const tf = require('@tensorflow/tfjs-node');

function normalized(data){ // i & r
    x1 = (data[0] - 12.585) / 6.813882
    x2 = (data[1] - 51.4795) / 29.151289
    x3 = (data[0] * 552.6264) + 650.4795
    x4 = (data[1] * 12153.8) + 10620.5615
    return [x1, x2, x3, x4]
}

const argFact = (compareFn) => (array)  => array.map((el,idx) => [el, idx]).reduce(compareFn)[1]
const argMax = argFact((min, el) => (el[0] > min[0] ? el:min))

function ArgMax(res){
  label = "NORMAL"
  cls_data = []
  for(i=0; i<res.length; i++){
    cls_data[i] = res[i]
  }
  console.log(cls_data,argMax(cls_data));
  
  if(argMax(cls_data) == 1){
    label  = "OVER VOLTAGE"
  }if(argMax(cls_data) == 0){
    label = "DROP VOLTAGE"
  }
  return label 
  
}


async function predict(data){
    let in_dim = 4;
    
    data = normalized(data);
    shape = [1, in_dim];

    tf_data = tf.tensor2d(data, shape);

    try{
        // path load in public access => github
        const path = 'https://raw.githubusercontent.com/achriziq/achriziq-jsta_riziq/main/public/cls_model/model.json';
        const model = await tf.loadGraphModel(path);
        
        predict = model.predict(
                tf_data
        );
        result = predict.dataSync();
        return denormalized( result );
        
    }catch(e){
      console.log(e);
    }
}

module.exports = {
    predict: predict 
}
  

