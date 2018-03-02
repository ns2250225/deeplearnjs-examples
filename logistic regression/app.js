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


// 散点图数据
const data_scatter1 = []
const data_scatter2 = []
for (let elem of data) {
    if (elem.x > 40 && elem.y > 40) {
        data_scatter1.push([elem.x, elem.y])
    } else {
        data_scatter2.push([elem.x, elem.y])
    }  
}
// 直线数据
const data_line = [
    [0, parseFloat((100 * -w_predict[0] - b_predict[0])/w_predict[1])],
    [100, parseFloat((0 * -w_predict[0] - b_predict[0])/w_predict[1])]
]
console.log(data_line)

// 绘出结果图形
var options = {
    title: {
        text: 'deeplearn.js的逻辑回归'                 
    },
    xAxis: {
        min: 0,
        max: 100
    },
    yAxis: {
        min: 0,
        max: 100
    },
    series: [
        {
            type: 'line',
            color: '#030303',
            data: data_line
        },  
        {
            type: 'scatter',
            marker: {
                symbol: 'cross',  
                radius: 4         
            },
            color: '#FF0000',
            data: data_scatter1
        },
        {
            type: 'scatter',
            marker: {
                symbol: 'cross',  
                radius: 4         
            },
            color: '#6B8E23',
            data: data_scatter2
        }
    ]
}
// 图表初始化函数
var chart = Highcharts.chart('container', options);




