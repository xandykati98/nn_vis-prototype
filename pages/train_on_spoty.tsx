import type { NextPage } from 'next'
import Head from 'next/head'
import Image from 'next/image'
import { useEffect, useRef, useState } from 'react'
import { LayerConfig, links_ws, NeuralNetwork, scale6, scaleBetween, setLinks, sigmoidRandom } from '../nn_organized'
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
  neuron_ids: number[],
  rede: any,
  showXY?: boolean,
}

const renderWs = (index:number, neuron_ids: number[], useful_ws: {
  neuron_id: number;
  weights: string[];
}[]) => {
  const [max_min, setMaxMin] = useState([0,0])
  useEffect(() => {
    setInterval(() => {
      // get max of links_ws
      let max = 0
      let min = 0
      for (const key in links_ws) {
        const ws = links_ws[key]
        if (ws > max) {
          max = ws
        }
        if (ws < min) {
          min = ws
        }
      }
      setMaxMin([max, min])

    }, 1000)
  }, [])
  if (index == 0)
  return;
  return <svg width={160} height={'200vh'} style={{marginTop: 120}} >
    <g className="links" width={160} height={200} >
      {
        neuron_ids.map((neuron_id, index) => {
          return <g width={160} height={200}>
            {
              useful_ws.find(({ neuron_id: n_id }) => n_id === neuron_id)?.weights.map((weights,w_index) => {
                const [from, to] = weights.split('_to_')
                const link = links_ws[weights]
                return [
                  <line className={"path "+(link<0?'reverse':'')} x1={0} y1={w_index * 220} x2={160} y2={index *220} stroke={`rgba(0,0,${from === to ? 255 : 0 },${scaleBetween(link, 0.1, 1, max_min[0], max_min[1])})`} strokeWidth={4} />,
                  <line onClick={() => alert(weights)} x1={0} y1={w_index * 220} x2={160} y2={index *220} stroke={`rgba(0,0,${from === to ? 255 : 0 },0.2)`} strokeWidth={4} />,
                  from === to && <text x={0} y={w_index * 220}>Bias!</text>,
                ].filter(_=>_)
              })
            }
          </g>
        })
      }
    </g>
  </svg>
}



const LayerComp = ({ neuron_ids, config, index, useful_ws, showXY }:LayerCompProps) => {

  return <div className={styles.layer}>
    <h3>{index !== 0 ? config?.activation_function || 'sigmoid': 'input'}</h3>
    <div style={{display: 'flex'}}>
      <>      
      {renderWs(index, neuron_ids, useful_ws)}
      <div className="neuron_maps">
        {neuron_ids.map(neuron_id => {
          const ws = useful_ws.find(w => w.neuron_id === neuron_id)?.weights
          return <div key={neuron_id}>
            <h4>{neuron_id}</h4>
            {
              ws ? 
              <NeuronVisualizer showXY={showXY} id={neuron_id} ws={ws.map(w => links_ws[w])}/>
              : <NeuronVisualizer id={neuron_id} showXY ws={[]}/>
            }
          </div>
        })}
      </div>
      </>
    </div>
  </div>
}

const boundary_size = 20;
const grid_min = -1;
const grid_max = 1;
let boundaries:{
  [key:number]: number[][]
} = {}






const nn = new NeuralNetwork()

nn.pushLayer({
  is_input: true,
  bias: true,
  neurons_number: 2+5
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
})

rede.neuron_ids = nn.layers.map(l => l.neurons.map(n => n.id))
// @ts-ignore
rede.layer_configs = nn.layers.map(l => l.config)

nn.createWeights()


function randUniform(a: number, b: number) {
  return Math.random() * (b - a) + a;
}

function genSpiral(deltaT: number, label: number, {noise = 0, n = 100}:{noise?: number, n?: number} = {}) {
  let points: {x:number,y:number,label:number}[] = [];
  for (let i = 0; i < n; i++) {
      let r = i / n * 1;
      let t = 1.75 * i / n * 2 * Math.PI + deltaT;
      let x = r * Math.sin(t) + randUniform(-1, 1) * noise;
      let y = r * Math.cos(t) + randUniform(-1, 1) * noise;
      points.push({x, y, label});
  }
  return points
}
const po = genSpiral(0, 1); // Positive examples.
const ne = genSpiral(Math.PI, -1); // Negative examples.

// Criação do conjunto de treinamento
let t_set:any[] = []

for (const { x, y, label} of [...po, ...ne]) {
  t_set.push({
    inputs: [x, y,
      x*y,
      x**2, 
      y**2, 
      Math.sin(x),
      Math.sin(y)],
    desired_outputs: [label]
  })
}

//while (t_set.length < 400) {
    // const input = [sigmoidRandom(-10, 10), sigmoidRandom(0, 10)];
    // const x = input[0];
    // const y = input[1];
    // 
    // const line = x**2;
    // t_set.push({
    //     inputs: input,
    //     desired_outputs: [((line) < y) ? 1 : 0]
    // })
    
//}

const train_config = {
  epochs: 215,
  momentum: 0.01,
  iteracoes: 1000,
  taxa_aprendizado: 0.04,
  training_set: t_set
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
            rede.neuron_ids.map((neuron_ids, index, arr) => {
              return <LayerComp
               showXY={index === arr.length-1}
               neuron_ids={neuron_ids} 
               index={index}
               useful_ws={neuron_ids.map(id => ({
                  neuron_id: id,
                  weights: Object.keys(links_ws).filter(key => key.endsWith(`_to_${id}`))
               })).filter(({weights}) => weights.length > 0)}
               config={rede.layer_configs[index]} 
               rede={rede}
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
  showXY?: boolean
}
const NeuronVisualizer = (props:NeuronVisualizerProps) => {
  
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const draw = (ctx:CanvasRenderingContext2D, frameCount: number, canvas:HTMLCanvasElement) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
    const rect = canvas!.getBoundingClientRect()
    // make a 10*10 squares in as a grid for the canvas
    const grid_size = boundary_size
    const grid_width = ctx.canvas.width / grid_size
    const grid_height = ctx.canvas.height / grid_size
    for (let i = 0; i < grid_size; i++) {
      for (let j = 0; j < grid_size; j++) {
        const x = i * grid_width
        const y = j * grid_height
        const x_in = scaleBetween(i, grid_min, grid_max, 0, boundary_size)
        const y_in = -1 * scaleBetween(j, grid_min, grid_max, 0, boundary_size)
        const opacity = scaleBetween(boundaries[props.id][i][j], 0, 1, -1, 1).toFixed(5)
        ctx.fillStyle = `rgba(0,0,0,${opacity})`
        ctx.fillRect(x, y, grid_width, grid_height)
      }
    }
    
    if (props.showXY) {
      for (const { inputs, desired_outputs } of t_set) {
        const [ x, y ] = inputs
        const [ desired ] = desired_outputs
        const x_scaled = scaleBetween(x, 0, ctx.canvas.width, -1, 1)
        const y_scaled = (-1 * scaleBetween(y, 0, ctx.canvas.height, -1, 1)) + ctx.canvas.height 
        ctx.fillStyle = desired === 1 ? `rgba(100,0,0,1)` : `rgba(0,100,0,1)`
        ctx.fillRect(x_scaled, y_scaled, 5, 5)
      }
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
      draw(context, frameCount, canvas as HTMLCanvasElement)
      animationFrameId = window.requestAnimationFrame(render)
    }
    render()
    
    return () => {
      window.cancelAnimationFrame(animationFrameId)
    }
  }, [draw])

  return <canvas ref={canvasRef} width={160} height={160}/>
}