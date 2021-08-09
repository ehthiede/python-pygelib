// Typedefs for convenience.
typedef SO3partA_CGproduct_cop SO3part_CGproduct;
typedef SO3partA_CGproduct_back0_cop SO3part_CGproduct_back0;
typedef SO3partA_CGproduct_back1_cop SO3part_CGproduct_back1;

// Routines for torch / GElib interface.
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
            dms.push_back(xr_size);
        }
        else if (i == ndim_real -2){
            l = (xr_size -1)/2;
        }
        else if (i == ndim_real-1){
            n = xr_size;
        }
    }
    Gdims gdms(dms);
    /* SO3partArray output(gdms, l, n, fill::noalloc, dev_real); */
    SO3partArray output(gdms, l, n, fill::noalloc, dev_real);

    // Set to view
    output.is_view=true;

    if(dev_real == 0){
        output.arr = x_real.data<float>();
        output.arrc = x_imag.data<float>();
    }
    else{
        output.arrg = x_real.data<float>();
        output.arrgc = x_imag.data<float>();
    }
    return output;
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

// CG Product Routines
inline void add_in_partArrayCGproduct(SO3partArray& output, const SO3partArray& x, const SO3partArray& y, const int offset){
    int l = output.getl();
    add_cellwise<SO3part_CGproduct>(output, x, y, offset);
}

inline void add_in_partArrayCGproduct_back0(SO3partArray& output, const SO3partArray& dy, const SO3partArray& b, const int offset){
    int l = output.getl();
    add_cellwise<SO3part_CGproduct_back0>(output, dy, b, offset);
}

inline void add_in_partArrayCGproduct_back1(SO3partArray& output, const SO3partArray& dy, const SO3partArray& a, const int offset){
    int l = output.getl();
    add_cellwise<SO3part_CGproduct_back1>(output, dy, a, offset);
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

// Miscellaneous and Testing Routines
inline SO3partArray partArrayCGproduct(const SO3partArray& x, const SO3partArray& y, const int l){
    return CGproduct(x, y, l);
}


void sampleprint(){
    SO3partArray cpu_array({3}, 2, 2, fill::ones, 0);
    SO3partArray gpu_array({3}, 2, 2, fill::ones, 1);
}



inline void sum_SO3partArrays_inplace(SO3partArray& x, const SO3partArray& y){
    x += y;
}


/* void joinedCGRoutine(vector<torch::Tensor> x_real, vector<torch::Tensor> x_imag, */
/*                      vector<torch::Tensor> y_real, vector<torch::Tensor> y_imag, */
/*                      vector<torch::Tensor> out_real, vector<torch::Tensor> out_imag){ */
void joinedCGRoutine(vector<torch::Tensor> x,
                     vector<torch::Tensor> y,
                     vector<torch::Tensor> out){
    int num_xs = x.size();
    int num_ys = y.size();
    int num_outs = out.size();

    /* cout << x[0].index({0}) << endl; */
    /* cout << x[0].index({1}) << endl; */
    
    /* for (int i=0; i< num_xs; i++){ */
    for (auto xi: x){
        SO3partArrayFromTensor(xi.index({0}), xi.index({1}));
    }

    for (auto yi: y){
        SO3partArrayFromTensor(yi.index({0}), yi.index({1}));
    }

    for (auto out_i: out){
        SO3partArrayFromTensor(out_i.index({0}), out_i.index({1}));
    }
    
    /* for (int i=0; i< num_ys; i++){ */
    /*     SO3partArrayFromTensor(y_real[i], y_imag[i]); */
    /* } */
    
    /* for (int i=0; i< num_outs; i++){ */
    /*     SO3partArrayFromTensor(out_real[i], out_imag[i]); */
    /* } */

}


void testGelibPtrs(vector<torch::Tensor> x_real, vector<torch::Tensor> x_imag,
                   vector<torch::Tensor> y_real, vector<torch::Tensor> y_imag,
                   vector<torch::Tensor> out_real, vector<torch::Tensor> out_imag){
    int num_xs = x_real.size();
    int num_ys = y_real.size();

    vector<SO3partArray*> x_parts;
    /* vector<SO3partArray*> y_parts; */

    /* for (int i=0; i< num_xs; i++){ */
    /*     SO3partArray* temp = SO3partArrayFromTensor(x_real[i], x_imag[i]); */
    /*     x_parts.push_back(temp); */
    /* } */

    /* vector<SO3partArray*> x_parts = {}; */
    for (int i=0; i< num_xs; i++){
        cout << "-------------__" << endl;
        SO3partArray temp(SO3partArrayFromTensor(x_real[i], x_imag[i]));
        SO3partArray temp2(SO3partArrayFromTensor(y_real[i], y_imag[i]));
        SO3partArray* temp_ptr = &temp;
        /* add_cellwise<SO3part_CGproduct>(output, x, y, offset); */
        cout << "i" << endl;
        x_parts.push_back(temp_ptr);
        
    }
    cout << "!!!!!!!!!!!!!!" << endl; 
    for (int i=0; i< num_xs; i++){
        cout << *x_parts[i] << endl;
    }
    /* for (int i=0; i< num_xs; i++){ */
    /*     SO3partArrayFromTensor(y_real[i], y_imag[i]); */
    /* } */
    
    /* for (int i=0; i< num_xs; i++){ */
    /*     SO3partArrayFromTensor(out_real[i], out_imag[i]); */
    /* } */


    /* for (int i=0; i< num_ys; i++) */
    /*     y_parts.push_back(SO3partArrayFromTensor(y_real[i], y_imag[i])); */
    /*     /1* y_parts[i] = SO3partArrayFromTensor(y_real[i], y_imag[i]); *1/ */
    
    /* cout << "_______" << endl; */
    /* for(int i=0; i < num_xs; i++){ */
    /*     SO3partArray x_i(*x_parts[i]); */
    /*     SO3partArray y_i(*y_parts[i]); */
    /*     x_i  += y_i; */
    /*     cout << x_i << x_parts[i] << endl; */
    /* } */
    
}


/* inline void rotate_SO3partArray(SO3partArray&x, const double phi, const double theta, const double psi){ */
/*     SO3element r(phi, theta, psi); */
/*     x.rotate(r); */
/* } */
