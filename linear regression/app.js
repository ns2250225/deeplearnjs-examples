// 初始化图形数据
const data = [] 
for (let i=0; i<100; i++) {
    data.push({
        x: Math.random() * 10,
        y: Math.random() * 10
    })
}
// 画出初始化散点图
const chart = new G2.Chart({
container: 'c1',    
width : 600,        
height : 300        
})
chart.source(data)
chart.point().position('x*y')
chart.render()


/**
 * 以下部分为deeplearn.js
 */
const x_list = []
const y_list = []

for (let elem of data) {
    x_list.push(elem.x)
    y_list.push(elem.y)
}

const x_data = dl.tensor1d(x_list)
const y_data = dl.tensor1d(y_list)

const Weights = dl.variable(dl.randomUniform([1]))
const Biases = dl.variable(dl.zeros([1]))

// 定义模型和损失函数
const f = x => Weights.mul(x).add(Biases)
const loss = (pred, label) => pred.sub(label).square().mean()
// 定义优化器，这里用sgd
const learningRate = 0.01
const optimizer = dl.train.sgd(learningRate)

// 训练模型
for (let i = 0; i < 10; i++) {
    optimizer.minimize(() => loss(f(x_data), y_data))
}


// 得出预测后的Weights和Biases
const w_predict = Weights.dataSync()
const b_predict = Biases.dataSync()

// 绘出结果直线
line_data = [
    {
        x: 0,
        y: 0 * w_predict + b_predict
    },
    {
        x: 10,
        y: 10 * w_predict + b_predict
    },
    data
]

const chart2 = new G2.Chart({
    container: 'c1',    
    width : 600,        
    height : 300        
})

chart2.source(line_data)
chart2.line().position('x*y')
chart2.render();




