import 'jimp';
import * as ort from "onnxruntime-web/webgpu";

export class InferenceSessionWGPU {
  constructor(width, height) {
    return (async () => {
      this.width = width;
      this.height = height;
      this.inputBuffers = [null, null, null, null, null];
      this.inputTensors = [null, null, null, null, null];
      this.outputBuffers = [null, null, null, null, null, null, null];
      this.outputTensors = [null, null, null, null, null, null, null];

      try {
        this.session = await ort.InferenceSession.create(`./noof${width}.onnx`,{
          executionProviders: ['webgpu'], preferredOutputLocation: "gpu-buffer"
        });
        this.device = ort.env.webgpu.device;
      } catch (e) {
          console.log(e);
          alert("Unable to create inference session!");
      }
      
      this.inputBuffers[0] = this.device.createBuffer({
        label: `InferenceSessionWGPU.inputBuffer0`,
        size: this.width * this.height * 3 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      this.inputTensors[0] = ort.Tensor.fromGpuBuffer(this.inputBuffers[0], {
        dataType: 'float32',
        dims: [1, 3, this.height, this.width]
      });
      this.outputBuffers[0] = this.device.createBuffer({
        label: `InferenceSessionWGPU.outputBuffer0`,
        size: this.width * this.height * 3 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.outputTensors[0] = ort.Tensor.fromGpuBuffer(this.outputBuffers[0], {
        dataType: 'float32',
        dims: [1, 3, this.height, this.width]
      });
      this.outputBuffers[1] = this.device.createBuffer({
        label: `InferenceSessionWGPU.outputBuffer1`,
        size: this.width * this.height * 3 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      });
      this.outputTensors[1] = ort.Tensor.fromGpuBuffer(this.outputBuffers[1], {
        dataType: 'float32',
        dims: [1, 3, this.height, this.width]
      });


      this.architecture = [32, 64, 64, 80, 96];
      for (var i = 0; i < this.inputBuffers.length - 1; i++) {
        var w = this.width / 2**i;
        var h = this.height / 2**i;
        this.inputBuffers[this.inputBuffers.length - 1 - i] = this.device.createBuffer({
          label: `InferenceSessionWGPU.inputBuffer${this.inputBuffers.length - 1 - i}`,
          size: this.architecture[i] * w * h * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }); 
        this.inputTensors[this.inputBuffers.length - 1 - i] = ort.Tensor.fromGpuBuffer(
          this.inputBuffers[this.inputBuffers.length - 1 - i], 
          {
            dataType: 'float32',
            dims: [1, this.architecture[i], h, w]
          }          
        );
      }

      for (var i = 0; i < this.outputBuffers.length - 2; i++) {
        var w = this.width / 2**i;
        var h = this.height / 2**i;
        this.outputBuffers[this.outputBuffers.length - 1 - i] = this.device.createBuffer({
          label: `InferenceSessionWGPU.outputBuffer${this.outputBuffers.length - 1 - i}`,
          size: this.architecture[i] * w * h * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }); 
        this.outputTensors[this.outputBuffers.length - 1 - i] = ort.Tensor.fromGpuBuffer(
          this.outputBuffers[this.outputBuffers.length - 1 - i], 
          {
            dataType: 'float32',
            dims: [1, this.architecture[i], h, w]
          }          
        );
      }
      return this;
    })();
  }

  async release() {
    for (var i = 0; i < this.inputBuffers.length; i++) {
      this.inputBuffers[i].destroy();
    }
    for (var i = 0; i < this.outputBuffers.length; i++) {
      this.outputBuffers[i].destroy();
    }
    await this.session.release();
  }

  cleanRecurrentState() {
    for (var i = 0; i < this.inputBuffers.length - 1; i++) {
      this.inputBuffers[this.inputBuffers.length - 1 - i].destroy();
      var w = this.width / 2**i;
      var h = this.height / 2**i;
      this.inputBuffers[this.inputBuffers.length - 1 - i] = this.device.createBuffer({
        label: `InferenceSessionWGPU.inputBuffer${this.inputBuffers.length - 1 - i}`,
        size: this.architecture[i] * w * h * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      }); 
      this.inputTensors[this.inputBuffers.length - 1 - i] = ort.Tensor.fromGpuBuffer(
        this.inputBuffers[this.inputBuffers.length - 1 - i], 
        {
          dataType: 'float32',
          dims: [1, this.architecture[i], h, w]
        }          
      );
    }
  }

  async runInference() {
    // create feeds with the input name from model export and the tensors
    const feeds = {};
    for (var i = 0; i < this.inputBuffers.length; i++) {
      feeds[this.session.inputNames[i]] = this.inputTensors[i];
    }
    const fetches = {};
    for (var i = 0; i < this.outputBuffers.length; i++) {
      fetches[this.session.outputNames[i]] = this.outputTensors[i];
    }
    const start = performance.now();  
    await this.session.run(feeds, fetches);

    // var readInput = this.device.createBuffer({
    //     size: this.outputBuffers[4].size,
    //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    // });
    // var commandEncoder = this.device.createCommandEncoder();
    // commandEncoder.copyBufferToBuffer(
    //     this.outputBuffers[4], 0, readInput, 0,
    //     readInput.size);
    // this.device.queue.submit([commandEncoder.finish()]);
    // await this.device.queue.onSubmittedWorkDone();
    // await readInput.mapAsync(GPUMapMode.READ);
    // var test = new Float32Array(readInput.getMappedRange());
    // console.log(test);
    // for (var i = 0; i < test.length; i++) {
    //   if (test[i] != 0) {
    //     console.log(i);
    //   }
    // }

    // copy recurrent state back into input
    var commandEncoder = this.device.createCommandEncoder();
    for (var i = 1; i < this.inputBuffers.length; i++) {
      commandEncoder.copyBufferToBuffer(
        this.outputBuffers[i + 2], 0, this.inputBuffers[i], 0, this.outputBuffers[i + 2].size
      );
    }
    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const end = performance.now();

    const inferenceTime = Math.round(end - start);
    return inferenceTime;
  }
  
};


 