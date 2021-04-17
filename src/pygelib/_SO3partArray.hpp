// Additional routines for interfacing with pytorch.
SO3partArray SO3partArrayFromTensor(const torch::Tensor& x_real, const torch::Tensor& x_imag){
    /* AT_ASSERT(x_real.dim() == 2,"SO3parts must be two-dimensional"); */
    int dev_real = int(x_real.type().is_cuda());
    int dev_imag = int(x_imag.type().is_cuda());
    int ndim_real = x_real.dim();
    int ndim_imag = x_imag.dim();
    AT_ASSERT(dev_real == dev_imag, "Real and imaginary tensors are on different devices.");
    AT_ASSERT(ndim_real == ndim_imag, "Real and imaginary tensor have a different number of indices.");

    int l;
    int n;
    vector<int> dms = {};
    for (int i = 0; i < ndim_real;  i++){
        int xr_size = x_real.size(i);
        AT_ASSERT(xr_size == x_imag.size(i),
                  "Sizes between real and imaginary tensor must agree.");
        if (i < ndim_real - 2){
            /* cout << i << endl; */
            /* cout << xr_size << endl; */
            dms.push_back(xr_size);
        }
        else if (i == ndim_real -2){
            l = (xr_size -1)/2;
            /* cout << i  << "l" << l << endl; */
        }
        else if (i == ndim_real-1){
            n = xr_size;
            /* cout << i  << "n" << n << endl; */
        }
    }
    Gdims gdms(dms);
    SO3partArray output(gdms, l, n, fill::noalloc, dev_real);

    // Set to view
    output.is_view=true;

    cout << "Tensor Pointer:" << x_real.data<float>() << endl;;
    if(dev_real == 0){
        cout << "is cpu" << endl;
        output.arr = x_real.data<float>();
        output.arrc = x_imag.data<float>();
        /* cout << "pointers are being set to :" << output.arr << output.arrc << endl; */
    }
    else{
        output.arrg = x_real.data<float>();
        output.arrgc = x_imag.data<float>();
    }

    /* SO3partArray ytst(gdms, l, n, fill::ones, dev_real); */
    /* output += ytst; */
    cout << "pointers before return" << output.arr << ", " << output.arrc << endl;

    return output;
}


pair<torch::Tensor, torch::Tensor> MoveSO3partArrayToTensor(SO3partArray& partarray){
    Gdims adms = partarray.adims;
    Gdims cdms = partarray.cdims;

    int num_adms = adms.size();
    int num_cdms = cdms.size();

    // vector<int64_t> v(num_adms + num_cdms);
    // for(int i=0; i<num_adms; i++)
    //     v[i]=adms[i];

    // for(int i=0; i<num_cdms; i++)
    //     v[i+num_adms]=cdms[i];

    vector<int64_t> v(num_adms + 1);
    for(int i=0; i<num_adms; i++)
        v[i]=adms[i];

    float flattened_cdm = 1;
    for(int i=0; i<num_cdms; i++)
        flattened_cdm *= float(cdms[i]);
        /* v[i+num_adms]=cdms[i]; */

    v[num_adms] = 32 * int(ceil(flattened_cdm / 32));


    torch::Tensor output_real;
    torch::Tensor output_imag;

    cout << "STarting recosntruction " << endl;
    if(partarray.dev == 0){
        auto options =
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .device(torch::kCPU)
                .requires_grad(false);

        output_real = torch::from_blob(partarray.arr, v, options);
        output_imag = torch::from_blob(partarray.arrc, v, options);
        partarray.arr = nullptr;
        partarray.arrc = nullptr;
    }
    else{
        Gdims gdms({5});
        vector<int64_t> tensordim({5, 3, 32});
        auto options =
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .device(torch::kCUDA)
                /* .device(torch::kCPU) */
                .requires_grad(false);
        cout << "defined options" << endl;
        /* SO3partArray test(gdms, 1, 32, fill::zero, 1); */
        /* /1* device deva=deviceid::CPU; *1/ */
        /* SO3partArray test2(test, 0); */
        /* output_real = torch::from_blob(test2.arr, tensordim, options0); */
        /* SO3partArray (gdms, 1, 32, fill::gaussian, 0); */
        /* output_real = torch::from_blob(partarray.arr, tensordim, options); */


        output_real = torch::from_blob(partarray.arrg, v, options);

        /* cout << "made first real tensor" << endl; */
        output_imag = torch::from_blob(partarray.arrgc, v, options);
        partarray.arrg = nullptr;
        partarray.arrgc = nullptr;
        /* abalaaba; */

    }

    cout << "Made device I think" << endl;
    /* torch::Tensor output = torch::randn(v, options); */
    pair<torch::Tensor, torch::Tensor> output(output_real, output_imag);
    return output;

    /* return torch::CPU(at::kFloat).tensorFromBlob(arr,v, [](void* data){delete reinterpret_cast<TYPE*>(data);}); */
}

// Utility functions for introspecting SO3partArrays
inline vector<int64_t> get_shape(const SO3partArray& x){
    int num_cdms = x.cdims.size();
    int num_adms = x.adims.size();
    vector<int64_t> v(num_adms + num_cdms);
    for(int i=0; i<num_adms; i++)
        v[i]=x.adims[i];
    for(int i=0; i<num_cdms; i++)
        v[i+num_adms]=x.cdims[i];
    return v;
}

inline int get_num_adims(const SO3partArray&x){
    return x.adims.size();
}

void sampleprint(){
    SO3partArray cpu_array({3}, 2, 2, fill::ones, 0);
    cout << "cpu_array" << endl;
    cout << cpu_array << endl;
    SO3partArray gpu_array({3}, 2, 2, fill::ones, 1);
    cout << "gpu_array" << endl;
    cout << gpu_array << endl;
}




inline void add_in_partArrayCGproduct(SO3partArray& output, const SO3partArray& x, const SO3partArray& y){
    int l = output.getl();
    output += CGproduct(x, y, l);
}

inline void add_in_partArrayCGproduct_back0(SO3partArray& output, const SO3partArray& x, const SO3partArray& other){
    int l = output.getl();
    output.add_CGproduct_back0(other, x, l);
}

inline void add_in_partArrayCGproduct_back1(SO3partArray& output, const SO3partArray& x, const SO3partArray& other){
    int l = output.getl();
    output.add_CGproduct_back1(x, other, l);
}

inline void sum_SO3partArrays_inplace(SO3partArray& x, const SO3partArray& y){
    cout << "STTTTAAAAAARRRRRTTTTTIIIIINNNNNGGGGGG" << endl;
    cout << "---------------------------" << endl;
    cout << "x ptr:" << x.arr <<  x.arrc << endl;
    /* cout << x.str("") << endl; */
    x += y;
    /* cout << x.str("") << endl; */
    cout << "x ptr:" << x.arr << x.arrc << endl;
    cout << "---------------------------" << endl;
}

inline SO3partArray partArrayCGproduct(const SO3partArray& x, const SO3partArray& y, const int l){
    return CGproduct(x, y, l);
}
