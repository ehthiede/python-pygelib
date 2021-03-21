// Additional routines for interfacing with pytorch.

SO3part SO3partFromTensor(const torch::Tensor& x_real, const torch::Tensor& x_imag){
    //AT_ASSERT(x.dim()==k,"Number of dimensions does not match.");
    AT_ASSERT(x_real.dim() == 2,"SO3parts must be two-dimensional");
    int mlength = x_real.size(0);
    int dev_real = int(x_real.type().is_cuda());
    int dev_imag = int(x_imag.type().is_cuda());
    AT_ASSERT((mlength % 2) == 1, "The first, SO3 Covariant dimension, must be odd");
    AT_ASSERT(x_real.size(0) == x_imag.size(0), "Sizes between real and imaginary tensor must agree.");
    AT_ASSERT(x_real.size(1) == x_imag.size(1), "Sizes between real and imaginary tensor must agree.");
    AT_ASSERT(dev_real == dev_imag, "Real and imaginary tensors are on different devices.");

    int l = (mlength - 1) / 2;
    x_real.contiguous();
    x_imag.contiguous();

    SO3partA data(l, x_real.size(1), -1, dev_real);
    data.arr = x_real.data<float>();
    data.arrc = x_imag.data<float>();
    SO3part output(data);
    return output;


/*     arr=x.data<TYPE>(); */
/*     is_view=true; */
}


