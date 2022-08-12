import type { NextPage } from 'next'
import Head from 'next/head'
import Image from 'next/image'
import { useEffect, useRef, useState } from 'react'
import { LayerConfig, links_ws, NeuralNetwork, scale6, scaleBetween, setLinks } from '../nn_organized'
import styles from '../styles/Home.module.css'
import rede from '../rede.json'

type LayerCompProps = {
  useful_ws: {
    neuron_id: number
    weights: string[]
  }[],
  index:number, 
  config: {
    [key: string]: any
    activation_function?: string
  },
  neuron_ids: number[]
}
const LayerComp = ({ neuron_ids, config, index, useful_ws }:LayerCompProps) => {
  return <div className={styles.layer}>
    <h3>{index !== 0 ? config?.activation_function || 'sigmoid': 'input'}</h3>
    <div>
      {neuron_ids.map(neuron_id => {
        const ws = useful_ws.find(w => w.neuron_id === neuron_id)?.weights
        return <div key={neuron_id}>
          <h4>{neuron_id}</h4>
          {
            ws ? 
            <NeuronVisualizer id={neuron_id} ws={ws.map(w => links_ws[w])}/>
            : null
          }
        </div>
      })}
    </div>
  </div>
}

const boundary_size = 25;
const grid_min = -6;
const grid_max = 6;
let boundaries:{
  [key:number]: number[][]
} = {}




type Example2D = {
  x: number,
  y: number,
  label: number
}
let n = 100;
const noise = 0.001;

/**
* Returns a sample from a uniform [a, b] distribution.
* Uses the seedrandom library as the random generator.
*/
function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}
function genSpiral(deltaT: number, label: number) {
  let points: Example2D[] = [];
  for (let i = 0; i < n; i++) {
      let r = i / n * 0.4;
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
  }
  return points
}

let po = genSpiral(0, 1); // Positive examples.
let ne = genSpiral(Math.PI, -1); // Negative examples.


const nn = new NeuralNetwork()

nn.pushLayer({
  is_input: true,

  neurons_number: 7,
})
nn.pushLayer({
  neurons_number:9,
  inline_bias: true,
  activation_function: 'tanh'
})
nn.pushLayer({
  neurons_number:8,
  inline_bias: true,
  activation_function: 'tanh'
})
nn.pushLayer({
  neurons_number:8,
  inline_bias: true,
  activation_function: 'tanh'
})
nn.pushLayer({
  neurons_number:8,
  inline_bias: true,
  activation_function: 'tanh'
})
nn.pushLayer({
  neurons_number:1,
  is_output: true,
  activation_function: 'tanh'
})

rede.neuron_ids = nn.layers.map(l => l.neurons.map(n => n.id))
// @ts-ignore
rede.layer_configs = nn.layers.map(l => l.config)
nn.createWeights()

const train_set = [...po, ...ne].map(sample => {
  return {
      inputs: [
          sample.x, // scaleBetween(sample.x, -1, 1, -8, 6),
          sample.y, // scaleBetween(sample.y, -1, 1, -8, 5),
          sample.x*sample.y, // scaleBetween(sample.x*sample.y, -1, 1, -(8*8), 5*6),
          // // weird
          sample.x**2, // scaleBetween(sample.x**2, -1, 1, 0, 6**2), 
          // // weird
          sample.y**2, // scaleBetween(sample.y**2, -1, 1, 0, 5**2),
          Math.sin(sample.x), 
          Math.sin(sample.y)
      ],
      desired_outputs: [sample.label],
  }
});
const input_mins = train_set.reduce((acc, sample) => {
  return {
      x: Math.min(acc.x, sample.inputs[0]),
      y: Math.min(acc.y, sample.inputs[1]),
      xy: Math.min(acc.xy, sample.inputs[2]),
      x2: Math.min(acc.x2, sample.inputs[3]),
      y2: Math.min(acc.y2, sample.inputs[4]),
      sin: Math.min(acc.sin, sample.inputs[5]),
      sin2: Math.min(acc.sin2, sample.inputs[6]),
  }
}, {
  x: Infinity,
  y: Infinity,
  xy: Infinity,
  x2: Infinity,
  y2: Infinity,
  sin: Infinity,
  sin2: Infinity,
});
const input_maxs = train_set.reduce((acc, sample) => {
  return {
      x: Math.max(acc.x, sample.inputs[0]),
      y: Math.max(acc.y, sample.inputs[1]),
      xy: Math.max(acc.xy, sample.inputs[2]),
      x2: Math.max(acc.x2, sample.inputs[3]),
      y2: Math.max(acc.y2, sample.inputs[4]),
      sin: Math.max(acc.sin, sample.inputs[5]),
      sin2: Math.max(acc.sin2, sample.inputs[6]),
  }
}, {
  x: -Infinity,
  y: -Infinity,
  xy: -Infinity,
  x2: -Infinity,
  y2: -Infinity,
  sin: -Infinity,
  sin2: -Infinity,
});

const scaled_train_set = train_set.map(sample => {
  return {
      inputs: [
          scaleBetween(sample.inputs[0], -1, 1, input_mins.x, input_maxs.x),
          scaleBetween(sample.inputs[1], -1, 1, input_mins.y, input_maxs.y),
          scaleBetween(sample.inputs[2], -1, 1, input_mins.xy, input_maxs.xy),
          scaleBetween(sample.inputs[3], -1, 1, input_mins.x2, input_maxs.x2),
          scaleBetween(sample.inputs[4], -1, 1, input_mins.y2, input_maxs.y2),
          scaleBetween(sample.inputs[5], -1, 1, input_mins.sin, input_maxs.sin),
          scaleBetween(sample.inputs[6], -1, 1, input_mins.sin2, input_maxs.sin2),
      ],
      desired_outputs: [sample.desired_outputs[0]],
  }
});

const train_config = {
  epochs: 3115,
  momentum: 0.04,
  iteracoes: 500,
  taxa_aprendizado: 0.04,
  training_set: scaled_train_set
}

for (const layer_neurons of rede.neuron_ids) {
  for (const neuron_id of layer_neurons) {
    boundaries[neuron_id] = []
    // make a boundary_size x boundary_size grid 
    for (let i = 0; i < boundary_size; i++) {
      boundaries[neuron_id].push(new Array(boundary_size).fill(0))
    }
  }
}
const update_bondaries = () => {
  // loop through the boundary_size x boundary_size grid
  for (let i = 0; i < boundary_size; i++) {
    for (let j = 0; j < boundary_size; j++) {
      const x = scaleBetween(i, grid_min, grid_max, 0, boundary_size)
      const y = -1 * scaleBetween(j, grid_min, grid_max, 0, boundary_size)
      const { layer_neuron_outputs, output_response } = nn.guess([
        x,
        y,
        x*y,
        x**2, 
        y**2, 
        Math.sin(x),
        Math.sin(y)
      ])

      for (const neuron_outputs of layer_neuron_outputs) {
        for (const neuron_output of neuron_outputs) {
          boundaries[Number(neuron_output.origin.id)][i][j] = neuron_output.value;
        }
      }
      for (const neuron_output of output_response) {
        boundaries[Number(neuron_output.origin.id)][i][j] = neuron_output.value;
      }
    }
  }
}
const Home: NextPage = () => {
  
  useEffect(() => {
    // @ts-ignore
    window.nn = nn
    update_bondaries()
    nn.train({
      ...train_config,
      on_epoch_end: async (epoch, error) => {
        update_bondaries()
      }
    });
  }, [])

  return (
    <div className={styles.container}>
      <Head>
        <title>Create Next App</title>
        <meta name="description" content="Generated by create next app" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className={styles.main}>
        <h1 className={styles.subtitle}>
          Bem vindo ao Visualizador de MLP
        </h1>

        <p className={styles.description}>
          Rede neural com id {' '}
          <code className={styles.code}>0</code>
        </p>

        <div className={styles.layerContain}>
          {
            rede.neuron_ids.map((neuron_ids, index) => {
              return <LayerComp
               neuron_ids={neuron_ids} 
               index={index}
               useful_ws={neuron_ids.map(id => ({
                  neuron_id: id,
                  weights: Object.keys(links_ws).filter(key => key.endsWith(`_to_${id}`))
               })).filter(({weights}) => weights.length > 0)}
               config={rede.layer_configs[index]} 
               key={'LayerComp'+index}
              />
            })
          }
        </div>
      </main>

    </div>
  )
}

export default Home

type NeuronVisualizerProps = {
  ws: number[]
  id: number
}
const NeuronVisualizer = (props:NeuronVisualizerProps) => {
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const draw = (ctx:CanvasRenderingContext2D, frameCount: number) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    // make a 10*10 squares in as a grid for the canvas
    const grid_size = boundary_size
    const grid_width = ctx.canvas.width / grid_size
    const grid_height = ctx.canvas.height / grid_size
    for (let i = 0; i < grid_size; i++) {
      for (let j = 0; j < grid_size; j++) {
        const x = i * grid_width
        const y = j * grid_height
        ctx.fillStyle = `rgba(${255*((boundaries[props.id][i][j]-1)/-2)},${255*((boundaries[props.id][i][j]+1)/2)},0,1)`
        ctx.fillRect(x, y, grid_width, grid_height)
      }
    }
    for (const { inputs, desired_outputs } of scaled_train_set) {
      const [ x, y ] = inputs
      const [ desired ] = desired_outputs
      const x_in = scaleBetween(x, 0, ctx.canvas.width, input_mins.x, input_maxs.x)
      const y_in = (-1 * scaleBetween(y, 0, ctx.canvas.height, input_mins.y, input_maxs.y)) + ctx.canvas.height
      ctx.fillStyle = desired === -1 ? `rgba(100,0,0,1)` : `rgba(0,100,0,1)`
      ctx.fillRect(x_in, y_in, 5, 5)
    }
  }
  useEffect(() => {
    const canvas = canvasRef!.current
    const context = canvas!.getContext('2d') as CanvasRenderingContext2D;

    canvas!.onclick = (event) => {
      // get x and y from the grid based on the click
      const rect = canvas!.getBoundingClientRect()
      const x = (event.clientX - rect.left) / rect.width * boundary_size
      const y = (event.clientY - rect.top) / rect.height * boundary_size
      const i = Math.floor(x)
      const j = Math.floor(y)
      
      const x_in = scaleBetween(i, grid_min, grid_max, 0, boundary_size)
      const y_in = -1 * scaleBetween(j, grid_min, grid_max, 0, boundary_size)
      // get the output from the network
      console.log({i, j}, {x_in, y_in}, boundaries[props.id][i][j].toFixed(5))
    }

    let frameCount = 0
    let animationFrameId:number;
    
    //Our draw came here
    const render = () => {
      frameCount++
      draw(context, frameCount)
      animationFrameId = window.requestAnimationFrame(render)
    }
    render()
    
    return () => {
      window.cancelAnimationFrame(animationFrameId)
    }
  }, [draw])

  return <canvas ref={canvasRef} width={200} height={200}/>
}