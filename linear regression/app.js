
// 初始化图形数据
const data1 = [] 
for (let i=0; i<100; i++) {
    let tmp_x = Math.random() * 10
    let offset = Math.random() * 10
    let tmp_y = tmp_x * 3 + offset
    data1.push({
        x: tmp_x,
        y: tmp_y
    })
}


/**
 * 以下部分为deeplearn.js
 */
const x_list = []
const y_list = []

for (let elem of data1) {
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
for (let i = 0; i < 100; i++) {
    optimizer.minimize(() => loss(f(x_data), y_data))
}


// 得出预测后的Weights和Biases
const w_predict = Weights.dataSync()
const b_predict = Biases.dataSync()

// 散点图数据
const data_scatter = []
for (let elem of data1) {
    data_scatter.push([elem.x, elem.y])
}
// 直线数据
const data_line = [
    [0, parseFloat(0 * w_predict + b_predict)],
    [10, parseFloat(10 * w_predict + b_predict)]
]

// 绘出结果直线-散点图
var options = {
    title: {
        text: 'deeplearn.js的线性回归'                 
    },
    xAxis: {
        min: 0,
        max: 10
    },
    yAxis: {
        min: 0,
        max: 60
    },
    series: [
        {
            type: 'line',
            data: data_line
        },  
        {
            type: 'scatter',
            marker: {
                symbol: 'cross',  
                radius: 4         
            },
            data: data_scatter
        }
    ]
}
// 图表初始化函数
var chart = Highcharts.chart('container', options);

                    



