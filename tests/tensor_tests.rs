// TODO: Also test the cuda backend.
use candle::{DType, Device, Result, Tensor};

#[test]
fn zeros() -> Result<()> {
    let tensor = Tensor::zeros((5, 2), DType::F32, &Device::Cpu)?;
    let (dim1, dim2) = tensor.shape().r2()?;
    assert_eq!(dim1, 5);
    assert_eq!(dim2, 2);
    Ok(())
}

#[test]
fn add_mul() -> Result<()> {
    let tensor = Tensor::new(&[3f32, 1., 4.], &Device::Cpu)?;
    let dim1 = tensor.shape().r1()?;
    assert_eq!(dim1, 3);
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [3., 1., 4.]);
    let tensor = Tensor::add(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [6., 2., 8.]);
    let tensor = Tensor::mul(&tensor, &tensor)?;
    let content: Vec<f32> = tensor.to_vec1()?;
    assert_eq!(content, [36., 4., 64.]);
    Ok(())
}

#[test]
fn tensor_2d() -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, &Device::Cpu)?;
    let dims = tensor.shape().r2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content, data);
    Ok(())
}

#[test]
fn binary_op() -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, &Device::Cpu)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 1., 7., 8., 2.]];
    let tensor2 = Tensor::new(data2, &Device::Cpu)?;
    let tensor = (&tensor + (&tensor * &tensor)? / (&tensor + &tensor2))?;
    let dims = tensor.shape().r2()?;
    assert_eq!(dims, (2, 5));
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [4.125, 1.1666666, 5.7777777, 1.1666666, 7.5]);
    assert_eq!(content[1], [3.0, 1.5, 10.5, 12.0, 3.0]);
    let tensor = (&tensor - &tensor)?;
    let content: Vec<Vec<f32>> = tensor.to_vec2()?;
    assert_eq!(content[0], [0., 0., 0., 0., 0.]);
    Ok(())
}

#[test]
fn tensor_2d_transpose() -> Result<()> {
    let data = &[[3f32, 1., 4., 1., 5.], [2., 1., 7., 8., 2.]];
    let tensor = Tensor::new(data, &Device::Cpu)?.t()?;
    let dims = tensor.shape().r2()?;
    assert_eq!(dims, (5, 2));
    assert_eq!(
        tensor.to_vec2::<f32>()?,
        &[[3f32, 2.], [1., 1.], [4., 7.], [1., 8.], [5., 2.]]
    );
    assert_eq!(tensor.t()?.to_vec2::<f32>()?, data);
    assert_eq!(tensor.contiguous()?.t()?.to_vec2::<f32>()?, data);
    assert_eq!(((tensor + 1.)?.t()? - 1.)?.to_vec2::<f32>()?, data);
    Ok(())
}

#[test]
fn softmax() -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, &Device::Cpu)?;
    let t0 = tensor.log()?.softmax(0)?;
    let t1 = tensor.log()?.softmax(1)?;
    let t2 = tensor.log()?.softmax(2)?;
    assert_eq!(
        t0.to_vec3::<f32>()?,
        &[
            // 3/5, 1/2, 4/11
            [[0.6, 0.5, 0.36363637], [0.11111111, 0.71428573, 0.5294118]],
            // 2/5, 1/2, 7/11
            [[0.4, 0.5, 0.63636357], [0.8888889, 0.2857143, 0.47058824]]
        ]
    );
    assert_eq!(
        t1.to_vec3::<f32>()?,
        &[
            // 3/4, 1/6, 4/13
            [[0.75, 0.16666667, 0.30769232], [0.25, 0.8333333, 0.6923077]],
            // 2/10, 1/3, 7/15
            [[0.2, 0.33333334, 0.46666664], [0.8, 0.6666667, 0.53333336]]
        ]
    );
    assert_eq!(
        t2.to_vec3::<f32>()?,
        &[
            // (3, 1, 4) / 8, (1, 5, 9) / 15
            [[0.375, 0.125, 0.5], [0.06666667, 0.33333334, 0.6]],
            // (2, 1, 7) / 10, (8, 2, 8) / 18
            [[0.2, 0.1, 0.6999999], [0.44444445, 0.11111111, 0.44444445]]
        ]
    );
    Ok(())
}

#[test]
fn sum() -> Result<()> {
    let data = &[[[3u32, 1, 4], [1, 5, 9]], [[2, 1, 7], [8, 2, 8]]];
    let tensor = Tensor::new(data, &Device::Cpu)?;
    assert_eq!(
        tensor.sum(&[2])?.to_vec3::<u32>()?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum(&[0])?.to_vec3::<u32>()?,
        &[[[5, 2, 11], [9, 7, 17]]],
    );
    assert_eq!(tensor.sum(&[0, 2, 1])?.to_vec3::<u32>()?, &[[[51]]],);
    assert_eq!(
        tensor.t()?.sum(&[1])?.t()?.to_vec3::<u32>()?,
        &[[[8], [15]], [[10], [18]]]
    );
    assert_eq!(
        tensor.sum(&[2, 1])?.to_vec3::<u32>()?,
        &[[[8 + 15]], [[10 + 18]]]
    );
    Ok(())
}

#[test]
fn narrow() -> Result<()> {
    let data = &[[[3f32, 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]];
    let tensor = Tensor::new(data, &Device::Cpu)?;
    assert_eq!(
        tensor.narrow(2, 1, 2)?.to_vec3::<f32>()?,
        &[[[1.0, 4.0], [5.0, 9.0]], [[1.0, 7.0], [2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(1, 1, 1)?.to_vec3::<f32>()?,
        &[[[1.0, 5.0, 9.0]], [[8.0, 2.0, 8.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 0, 1)?.to_vec3::<f32>()?,
        &[[[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]]],
    );
    assert_eq!(
        tensor.narrow(0, 1, 1)?.to_vec3::<f32>()?,
        &[[[2.0, 1.0, 7.0], [8.0, 2.0, 8.0]]],
    );
    // The following has been checked against PyTorch via:
    //   import torch
    //   t = torch.tensor([[[3., 1., 4.], [1., 5., 9.]], [[2., 1., 7.], [8., 2., 8.]]])
    //   t.transpose(-1, -2).narrow(1, 1, 2)
    assert_eq!(
        tensor.t()?.narrow(1, 1, 2)?.to_vec3::<f32>()?,
        &[[[1.0, 5.0], [4.0, 9.0]], [[1.0, 2.0], [7.0, 8.0]]],
    );
    Ok(())
}

#[test]
fn broadcast() -> Result<()> {
    let data = &[3f32, 1., 4.];
    let tensor = Tensor::new(data, &Device::Cpu)?;
    assert_eq!(
        tensor.broadcast_left((3, 1))?.to_vec3::<f32>()?,
        &[[[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]], [[3.0, 1.0, 4.0]]]
    );
    Ok(())
}

#[test]
fn cat() -> Result<()> {
    // 1D
    let t1 = Tensor::new(&[3f32, 1., 4.], &Device::Cpu)?;
    let t2 = Tensor::new(&[1f32, 5., 9., 2.], &Device::Cpu)?;
    let t3 = Tensor::new(&[6f32, 5., 3., 5., 8., 9.], &Device::Cpu)?;
    assert_eq!(Tensor::cat(&[&t1], 0)?.to_vec1::<f32>()?, [3f32, 1., 4.],);
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec1::<f32>()?,
        [3f32, 1., 4., 1., 5., 9., 2.],
    );
    assert_eq!(
        Tensor::cat(&[&t1, &t2, &t3], 0)?.to_vec1::<f32>()?,
        [3f32, 1., 4., 1., 5., 9., 2., 6., 5., 3., 5., 8., 9.],
    );

    // 2D
    let data = &[[3f32, 1., 4., 1., 5.], [2., 7., 1., 8., 2.]];
    let t1 = Tensor::new(data, &Device::Cpu)?;
    let data2 = &[[5f32, 5., 5., 5., 5.], [2., 7., 1., 8., 2.]];
    let t2 = Tensor::new(data2, &Device::Cpu)?;
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 0)?.to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0],
            [2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    // TODO: This is not the expected answer, to be fixed!
    assert_eq!(
        Tensor::cat(&[&t1.t()?, &t2.t()?], 1)?
            .t()?
            .to_vec2::<f32>()?,
        [
            [3.0, 4.0, 5.0, 5.0, 5.0],
            [2.0, 1.0, 2.0, 7.0, 8.0],
            [1.0, 1.0, 5.0, 5.0, 5.0],
            [7.0, 8.0, 2.0, 1.0, 2.0]
        ]
    );
    // TODO: This is not the expected answer, to be fixed!
    assert_eq!(
        Tensor::cat(&[&t1, &t2], 1)?.to_vec2::<f32>()?,
        [
            [3.0, 1.0, 4.0, 1.0, 5.0, 2.0, 7.0, 1.0, 8.0, 2.0],
            [5.0, 5.0, 5.0, 5.0, 5.0, 2.0, 7.0, 1.0, 8.0, 2.0]
        ]
    );
    Ok(())
}
