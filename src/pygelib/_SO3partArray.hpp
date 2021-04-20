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

    /* cout << "Tensor Pointer:" << x_real.data<float>() << endl;; */
    if(dev_real == 0){
        /* cout << "is cpu" << endl; */
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
    /* cout << "pointers before return" << output.arr << ", " << output.arrc << endl; */

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

    /* cout << "STarting recosntruction " << endl; */
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
        auto options =
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .layout(torch::kStrided)
                .device(torch::kCUDA)
                /* .device(torch::kCPU) */
                .requires_grad(false);


        output_real = torch::from_blob(partarray.arrg, v, options);

        /* cout << "made first real tensor" << endl; */
        output_imag = torch::from_blob(partarray.arrgc, v, options);
        partarray.arrg = nullptr;
        partarray.arrgc = nullptr;
        /* abalaaba; */

    }

    /* cout << "Made device I think" << endl; */
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

inline int get_num_channels(const SO3partArray&x){
    int size = x.cdims.size();
    return x.cdims[size-1];
}

void sampleprint(){
    SO3partArray cpu_array({3}, 2, 2, fill::ones, 0);
    /* cout << "cpu_array" << endl; */
    /* cout << cpu_array << endl; */
    SO3partArray gpu_array({3}, 2, 2, fill::ones, 1);
    /* cout << "gpu_array" << endl; */
    /* cout << gpu_array << endl; */
}




inline void add_in_partArrayCGproduct(SO3partArray& output, const SO3partArray& x, const SO3partArray& y){
    int l = output.getl();
    output += CGproduct(x, y, l);
}

/* inline void add_in_partArrayCGproduct_back0(SO3partArray& output, const SO3partArray& x, const SO3partArray& other){ */
/*     int l = output.getl(); */
/*     output.add_CGproduct_back0(other, x, l); */
/* } */

/* inline void add_in_partArrayCGproduct_back1(SO3partArray& output, const SO3partArray& x, const SO3partArray& other){ */
/*     int l = output.getl(); */
/*     output.add_CGproduct_back1(x, other, l); */
/* } */

inline void sum_SO3partArrays_inplace(SO3partArray& x, const SO3partArray& y){
    x += y;
}

inline SO3partArray partArrayCGproduct(const SO3partArray& x, const SO3partArray& y, const int l){
    return CGproduct(x, y, l);
}

inline vector<int> estimate_num_products(const vector<int>& types_one, const vector<int>& types_two){
    SO3type tau1(types_one);
    SO3type tau2(types_two);
    SO3type product(CGproduct(types_one, types_two));

    int num_out = product.size();
    vector<int> v(num_out);
    for(int i=0; i<num_out; i++){
        v[i] = product[i];
    }
    return v;
}
