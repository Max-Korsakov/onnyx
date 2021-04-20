import { Component, AfterViewInit } from '@angular/core';
import { InferenceSession, Tensor } from "onnxjs";
import npyjs from 'npyjs'
import ndarray from 'ndarray'
import * as ndarrayops from 'ndarray-ops'


declare var cv

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements AfterViewInit {

  public onnxJsSession: InferenceSession
  public npy;
  public canvas: HTMLCanvasElement
  public context: CanvasRenderingContext2D
  public height: number
  public width: number
  public player1Landmarks: number[] = [135, 95, 166, 101, 148, 108, 126, 128, 159, 135]
  public player2Landmarks: number[] = [82, 75, 121, 76, 104, 96, 86, 121, 117, 122]
  public player3Landmarks = [341, 130, 370, 135, 340, 152, 338, 172, 358, 178]

  public player1ImageUrl: string = '../assets/player1.jpg'
  public player2ImageUrl: string = '../assets/player2.jpg'
  public player3ImageUrl: string = '../assets/player3.jpg'

  public onnxModelName: string = "arcfaceresnet100-8_updated"


  ngAfterViewInit() {
    this.canvas = document.getElementById('canvas') as HTMLCanvasElement;
    this.context = this.canvas.getContext('2d');
    this.npy = new npyjs();
    setTimeout(() => {
      this.loadImage(this.player3ImageUrl, this.context)
    }, 3000) // some timeout to be sure that opencv downloaded, should be improved later
  }


  loadImage(url: string, context: CanvasRenderingContext2D) {
    let image = new Image();
    image.src = url;
    image.onload = () => {
      this.canvas.height = image.height
      this.canvas.width = image.width
      context.drawImage(image, 0, 0);
      this.runScript()
    }
  }


  async loadModel(url) {
    await this.onnxJsSession.loadModel(url);
  }


  async runScript() {
    this.onnxJsSession = new InferenceSession({ backendHint: 'webgl' });
    const url = `../assets/${this.onnxModelName}.onnx`;
    await this.loadModel(url)

    let emb1 = this.calcImgEmbedding(this.canvas, this.player3Landmarks)
  }

  calcImgEmbedding(canvas, landmarks: number[]) {
    let img = cv.imread(canvas)
    let aligned_img = this.alignFace(img, landmarks)
    return this.get_embedding(this.onnxJsSession, aligned_img)
  }

  async get_embedding(rt_onnxJsSession, aligned) {
    console.log(aligned)
    // let input_batch = np.float32(aligned)
    // console.log('input_batch', aligned.shape)
    //let input_name = rt_onnxJsSession.get_inputs()[0].name
    // let array = this.cvMatToArray(aligned)
    // console.log('array', array)
    // let img = cv.imread(this.canvas)
    // let context = (this.canvas as any).getContext('2d');
    // const imgData = context.getImageData(0, 0, 112, 112);
    // console.log(imgData.data)
    // let a = new Array(...imgData.data)
    // let filtered = a.filter((item, i) => {
    //   if (!((i + 1) % 4 === 0)) {
    //     return item
    //   }
    // })
    // console.log(filtered, filtered.length)
    // let newOne = []
    // let imageData = new cv.Mat()
    // cv.cvtColor(aligned, imageData, cv.COLOR_RGBA2RGB);
    // cv.imshow(this.canvas, imageData)

    //  let gdInage = imageData.data
    // for (let i = 1; i <= gdInage.length; i += 3) {
    //   // do whatever you need to with the values and simply mutate the buffer instance :)
    //   //const [r, g, b] = pxlFunc(imageData[i - 1], imageData[i], imageData[i + 1]);
    //   newOne[i - 1] = gdInage[i - 1];
    //   newOne[i] = gdInage[i];
    //   newOne[i + 1] = gdInage[i + 1];
    // }

    // console.log(newOne, newOne.length)

    // console.log(imageData.data)

    let data = this.preprocess(aligned.data, 112, 112)
    // let ctx = this.canvas.getContext('2d')
    // const imgData = ctx.getImageData(0, 0, 112, 112);
    console.log(data)
    const inputs: any = new Tensor(data, 'float32', [1, 3, 112, 112])

    //inputs[0].internalTensor.integerData = true;

    //const inputs = [new Tensor(new Float32Array(data), 'float32')]
    console.log(inputs)
    let output
    try {

      output = await rt_onnxJsSession.run([inputs])//console.log(output)
      const outputData = output.values().next().value.data;
    } catch (e) {
      console.log(e)
    }

    //let embedding = output[0].flatten()
  }

  alignFace(img, srcLandmarks, detection_method = 'faceapi') {
    let outputSize = [112, 112]
    if (detection_method === 'faceapi') {

      let MatrixFromSrc = new cv.matFromArray(
        5,
        2,
        cv.CV_32F,
        srcLandmarks
      )
      let MatrixFromDst = new cv.matFromArray(
        5,
        2,
        cv.CV_32F,
        [
          38.2946, 51.6963,
          73.5318, 51.5014,
          56.0252, 71.7366,
          41.5493, 92.3655,
          70.7299, 92.2041]
      )

      let t_matrix = cv.estimateAffinePartial2D(MatrixFromSrc, MatrixFromDst)
      let dst = new cv.Mat();
      let dsize = new cv.Size(outputSize[0], outputSize[1]);
      cv.warpAffine(img, dst, t_matrix, dsize);
      cv.imshow(this.canvas, dst)
      return dst
    }
  }


  cvMatToArray(mat) {
    let size = mat.size();
    console.log('size', size)
    let arr = [];
    if (mat.type() === cv.CV_32SC2) {
      size.width = 2; // because we have 2 channels
    }
    console.log('type', mat.type())
    for (let i = 0; i < size.height; i++) {
      let row = [];
      for (let j = 0; j < size.width; j++) {
        let value;
        switch (mat.type()) {
          case cv.CV_8UC4: value = mat.ucharAt(i, j); break;
          case cv.CV_8S: value = mat.charAt(i, j); break;
          case cv.CV_16U: value = mat.ushortAt(i, j); break;
          case cv.CV_16S: value = mat.shortAt(i, j); break;
          case cv.CV_32S: value = mat.intAt(i, j); break;
          case cv.CV_32SC2: value = mat.intAt(i, j); break;
          case cv.CV_32F: value = mat.floatAt(i, j); break;
          case cv.CV_64F: value = mat.doubleAt(i, j); break;
          default: throw `Unknown mat data type ${mat.type()}`;
        }
        row.push(value);
      }
      arr.push(row);
    }
    return arr;
  }

  convertBlock(incomingData) { // incoming data is a UInt8Array
    var i, l = incomingData.length;
    console.log(i, l)
    var outputData = new Float32Array(incomingData.length);
    for (i = 0; i < l; i++) {
      outputData[i] = (incomingData[i] - 128) / 128.0;
    }
    return outputData;
  }

  preprocess(data, width, height) {

    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);
    console.log(dataProcessed)

    // Normalize 0-255 to (-1)-1
    ndarrayops.divseq(dataFromImage, 128.0);
    ndarrayops.subseq(dataFromImage, 1.0);

    // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
    ndarrayops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
    ndarrayops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ndarrayops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));
    console.log(dataProcessed)
    return dataProcessed.data;
  }

}
