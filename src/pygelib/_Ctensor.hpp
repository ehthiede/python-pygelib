// Additional routines for interfacing with pytorch.

Ctensor CtensorFromTensor(const torch::Tensor& x_real, const torch::Tensor& x_imag){
    //AT_ASSERT(x.dim()==k,"Number of dimensions does not match.");
    AT_ASSERT(x_real.dim() == 2,"SO3parts must be two-dimensional");
    int dev_real = int(x_real.type().is_cuda());
    int dev_imag = int(x_imag.type().is_cuda());
    int ndim_real = x_real.dim();
    int ndim_imag = x_imag.dim();
    AT_ASSERT(ndim_real == ndim_imag, "Real and imaginary tensor have a different number of indices.");

    vector<int> dms = {};
    for (int i = 0; i < ndim_real;  i++){
        int xr_size = x_real.size(i);
        AT_ASSERT(xr_size == x_imag.size(i),
                  "Sizes between real and imaginary tensor must agree.");
        dms.push_back(xr_size);
    }
    Gdims gdms(dms);

    AT_ASSERT(dev_real == dev_imag, "Real and imaginary tensors are on different devices.");

    x_real.contiguous();
    x_imag.contiguous();

    CtensorA data(gdms, -1, dev_real);
    data.arr = x_real.data<float>();
    data.arrc = x_imag.data<float>();
    Ctensor output(data);
    return output;


/*     arr=x.data<TYPE>(); */
/*     is_view=true; */
}



