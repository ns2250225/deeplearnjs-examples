// 创建 tensor 张量
function create_tensor() {
    console.log("创建基础tensor张量...")
    // dl.scalar(value[number|boolean], dtype[float32|int32|bool])
    dl.scalar(3.14).print();
    // dl.tensor1d(value[TypedArray|Array], dtype[float32|int32|bool])
    dl.tensor1d([1, 2, 3]).print();
    // dl.tensor2d(value[TypedArray|Array], shape[number,number], dtype[float32|int32|bool])
    dl.tensor2d([1, 2, 3, 4], [2, 2]).print();
    // dl.tensor3d(value[TypedArray|Array], shape[number,number,number], dtype[float32|int32|bool])
    dl.tensor3d([1, 2, 3, 4], [2, 2, 1]).print();
    // dl.tensor4d(value[TypedArray|Array], shape[number,number,number,number], dtype[float32|int32|bool])
    dl.tensor4d([1, 2, 3, 4], [1, 2, 2, 1]).print();
    // dl.variable(initialValue[tensor], trainable[bool], name[string], dtype[float32|int32|bool])
    dl.variable(dl.tensor([1, 2, 3])).print();
}


// tensor张量的类型转换
function transformations() {
    console.log("tensor的类型转换...")
    // 转换tensor的dtype
    var x = dl.tensor1d([1.5, 2.5, 3]);
    dl.cast(x, 'int32').print();

    // 增加tensor的秩
    var x = dl.tensor1d([1, 2, 3, 4]);
    const axis = 1;
    x.expandDims(axis).print();

    //
    var x = dl.tensor1d([1, 2, 3, 4]);
    x.pad([[1, 2]]).print();

    // 更改tensor的shape
    var x = dl.tensor1d([1, 2, 3, 4]);
    x.reshape([2, 2]).print();

    //
    var x = dl.tensor([1, 2, 3, 4], [1, 1, 4]);
    x.squeeze().print();
}

