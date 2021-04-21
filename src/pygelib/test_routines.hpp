/* inline SO3partArray init_tensors_and_take_product(){ */
/*     SO3partArray x({1}, 1, 32, fill::ones, 1); */
/*     SO3partArray y({1}, 7, 32, fill::ones, 1); */
/*     return CGproduct(x, y, 7); */
/* } */

inline void init_parts_and_take_product(){
    SO3partArray x({1}, 1, 32, fill::ones, 1);
    SO3partArray y({1}, 3, 32, fill::ones, 1);
    CGproduct(x, y, 3);
    return;
}

inline void init_parts_and_dont_take_product(){
    SO3partArray x({1}, 1, 32, fill::ones, 1);
    SO3partArray y({1}, 7, 32, fill::ones, 1);
    /* CGproduct(x, y, 7); */
    return;
}
