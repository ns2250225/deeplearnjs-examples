// 初始化图形数据, 构造二分类数据，1代表60分以上，0代表60分以下
const data = [] 
for (let i=0; i<100; i++) {
    let tmp_x = Math.random() * 100
    let tmp_y = Math.random() * 100
    data.push({
        x: tmp_x,
        y: tmp_y,
        c: tmp_x > 40 && tmp_y > 40 ? 1: 0
    })
}
// 画出初始化散点图
const chart = new G2.Chart({
container: 'c1',    
width : 600,        
height : 300        
})
chart.source(data)
chart.point().position('x*y').color('c')
chart.render()


/**
 * 以下部分为deeplearn.js
 */
const x_list = []
const y_list = []

for (let elem of data) {
    x_list.push([elem.x, elem.y])
    y_list.push(elem.c)
}

const x_data = dl.tensor2d(x_list).transpose()
const y_data = dl.tensor2d(y_list)

// 训练目标
const Weights = dl.variable(dl.zeros([1, 2]))
const Biases = dl.variable(dl.zeros([1]))

// 定义模型和损失函数
const f = x => dl.sigmoid(Weights.matMul(x).add(Biases))
const loss = (pred, label) => pred.sub(label).square().mean() 

// 定义优化器，这里用sgd
const learningRate = 0.001
const optimizer = dl.train.sgd(learningRate)

// 训练模型
for (let i = 0; i < 1000; i++) {
    optimizer.minimize(() => loss(f(x_data), y_data))
}


// 得出预测后的Weights和Biases
const w_predict = Weights.dataSync()
const b_predict = Biases.dataSync()

console.log(w_predict, b_predict)

// 绘出结果直线
const line_data = [
    {
        x: 0,
        y: (100 * -w_predict[0] - b_predict[0]) / w_predict[1]
    },
    {
        x: 100,
        y: (0 * -w_predict[0] - b_predict[0]) / w_predict[1]
    }
] 


const chart2 = new G2.Chart({
    container: 'c1',    
    width : 600,        
    height : 300        
})

chart2.source(line_data)
chart2.line().position('x*y')
chart2.render();




