// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use cudarc::driver::safe::CudaDevice;
use cudarc::nccl::safe::{Comm, Id};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::rc::Rc;
use std::sync::Arc;

mod model;
use model::{Config, Llama};

const MAX_SEQ_LEN: usize = 4096;
const DEFAULT_PROMPT: &str = r"
EDWARD:
I wonder how our princely father 'scaped,
Or whether he be 'scaped away or no
From Clifford's and Northumberland's pursuit:
Had he been ta'en, we should have heard the news;
Had he been slain, we should have heard the news;
Or had he 'scaped, methinks we should have heard
The happy tidings of his good escape.
How fares my brother? why is he so sad?

RICHARD:
I cannot joy, until I be resolved
Where our right valiant father is become.
I saw him in the battle range about;
And watch'd him how he singled Clifford forth.
Methought he bore him in the thickest troop
As doth a lion in a herd of neat;
Or as a bear, encompass'd round with dogs,
Who having pinch'd a few and made them cry,
The rest stand all aloof, and bark at him.
So fared our father with his enemies;
So fled his enemies my warlike father:
Methinks, 'tis prize enough to be his son.
See how the morning opes her golden gates,
And takes her farewell of the glorious sun!
How well resembles it the prime of youth,
Trimm'd like a younker prancing to his love!

EDWARD:
Dazzle mine eyes, or do I see three suns?

RICHARD:
Three glorious suns, each one a perfect sun;
Not separated with the racking clouds,
But sever'd in a pale clear-shining sky.
See, see! they join, embrace, and seem to kiss,
As if they vow'd some league inviolable:
Now are they but one lamp, one light, one sun.
In this the heaven figures some event.

EDWARD:
'Tis wondrous strange, the like yet never heard of.
I think it cites us, brother, to the field,
That we, the sons of brave Plantagenet,
Each one already blazing by our meeds,
Should notwithstanding join our lights together
And over-shine the earth as this the world.
Whate'er it bodes, henceforward will I bear
Upon my target three fair-shining suns.
";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value_t = 4)]
    num_shards: usize,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    dtype: Option<String>,
}


#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };

    let api = Api::new()?;

    let model_id = args
        .model_id
        .unwrap_or_else(|| "meta-llama/Llama-2-7b-hf".to_string());
    println!("loading the model weights from {model_id}");
    let revision = args.revision.unwrap_or("main".to_string());
    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let config_filename = api.get("config.json")?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let mut filenames = vec![];
    for rfilename in [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        let filename = api.get(rfilename)?;
        filenames.push(filename);
    }


    let num_shards = args.num_shards;

    let (request_sender, _request_receiver) = tokio::sync::broadcast::channel::<(Tensor, usize, tokio::sync::mpsc::Sender<candle::Result<Tensor>>)>(10);
    
    let mut tasks = vec![];
    let id = Id::new().unwrap();
    let ready = Arc::new(tokio::sync::Notify::new());
    for rank in 0..num_shards {

        let mut request_receiver = request_sender.subscribe();
        let filenames = filenames.clone();
        let config = config.clone();
        let ready = ready.clone();
        tasks.push(std::thread::spawn( move ||  {
            let device = CudaDevice::new(rank)?;
            let comm = Rc::new(Comm::from_rank(device, rank, num_shards, id).unwrap());
            println!("Rank {rank:?} spawned");
        
            let device = Device::new_cuda(rank)?;
            let cache = model::Cache::new(dtype, &config, &device)?;
        
            println!("building the model");
            let vb = unsafe {
                candle_nn::var_builder::ShardedSafeTensors::var_builder(&filenames, dtype, &device)?
            };
            let llama = Llama::load(vb, &cache, &config, comm)?;

            ready.notify_one();

            while let Ok((input, index_pos, respond))  = request_receiver.blocking_recv() {
                println!("recv request Rank {rank}");
                let out = if !input.device().same_device(&device) {
                    input.to_device(&device)
                } else {
                    Ok(input)
                }.and_then(|input| {
                    llama.forward(&input, index_pos)
                });
                tokio::spawn(async move {
                    let _ = respond.send(out).await;
                });
            }
            anyhow::Ok(())
        }));
    }


    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let mut ready_num = 0;

    while ready_num < num_shards {
        ready.notified().await;
        ready_num += 1;
    }

    println!("starting the inference loop");
    let mut logits_processor = LogitsProcessor::new(args.seed, args.temperature, args.top_p);
    let mut new_tokens = vec![];
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let device = Device::Cpu;
    for index in 0..args.sample_len {
        let start_gen = std::time::Instant::now();
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;

        let (tx, mut rx) = tokio::sync::mpsc::channel(num_shards);

        request_sender.send((input, index_pos, tx))?;

        let logits = match rx.recv().await {
            Some(out) => out?,
            None => {
                anyhow::bail!("recv failed")
            }
        };


        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        new_tokens.push(next_token);

        println!("> {:?}", start_gen.elapsed());
        println!(
            "{} token: {} '{}'",
            index + 1,
            next_token,
            tokenizer.decode(&[next_token], true).map_err(E::msg)?
        );
    }
    let dt = start_gen.elapsed();

    println!(
        "{} tokens generated ({} token/s)\n----\n{}\n----",
        args.sample_len,
        args.sample_len as f64 / dt.as_secs_f64(),
        tokenizer
            .decode(new_tokens.as_slice(), true)
            .map_err(E::msg)?
    );
    Ok(())
}
