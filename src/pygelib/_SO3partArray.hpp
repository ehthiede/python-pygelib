// Additional routines for interfacing with pytorch.

SO3partArray SO3partArrayFromTensor(const torch::Tensor& x_real, const torch::Tensor& x_imag){
    /* AT_ASSERT(x_real.dim() == 2,"SO3parts must be two-dimensional"); */
    int dev_real = int(x_real.type().is_cuda());
    int dev_imag = int(x_imag.type().is_cuda());
    int ndim_real = x_real.dim();
    int ndim_imag = x_imag.dim();
    AT_ASSERT(dev_real == dev_imag, "Real and imaginary tensors are on different devices.");
    AT_ASSERT(ndim_real == ndim_imag, "Real and imaginary tensor have a different number of indices.");

    x_real.contiguous();
    x_imag.contiguous();

    int l;
    int n;
    vector<int> dms = {};
    for (int i = 0; i < ndim_real;  i++){
        int xr_size = x_real.size(i);
        AT_ASSERT(xr_size == x_imag.size(i),
                  "Sizes between real and imaginary tensor must agree.");
        if (i < ndim_real - 2){
            cout << i << endl;
            cout << xr_size << endl;
            dms.push_back(xr_size);
        }
        else if (i == ndim_real -2){
            l = (xr_size -1)/2;
            cout << i  << "l" << l << endl;
        }
        else if (i == ndim_real-1){
            n = xr_size;
            cout << i  << "n" << n << endl;
        }
    }


    Gdims gdms(dms);

    SO3partArrayA data(gdms, l, n, fill::gaussian, dev_real);
	// IS THIS OK FROM A MEMORY PERSPECTIVE?
    data.arr = x_real.data<float>();
    data.arrc = x_imag.data<float>();
    SO3partArray output(data);

    cout << "-------------" << endl;
    cout << "outputA" << endl;
    cout << output << endl;
    cout << "-------------" << endl;
    /* SO3partArray data(gdms, -1, dev_real); */

    return output;
}


torch::Tensor TensorFromSO3partArray(const SO3partArray partarray){
    Gdims adms = partarray.adims;
    Gdims cdms = partarray.cdims;

    int num_adms = adms.size();
    int num_cdms = cdms.size();

    vector<int64_t> v(num_adms + num_cdms);
    for(int i=0; i<num_adms; i++)
        v[i]=adms[i];

    for(int i=0; i<num_cdms; i++)
        v[i+num_adms]=cdms[i];

    auto options =
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCPU)
            .requires_grad(false);

    torch::Tensor output = torch::from_blob(partarray.arr, v, options);
    /* torch::Tensor output = torch::randn(v, options); */
    return output;

    /* return torch::CPU(at::kFloat).tensorFromBlob(arr,v, [](void* data){delete reinterpret_cast<TYPE*>(data);}); */
}
