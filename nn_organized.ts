/**
 * Objeto contendo valores dos pesos das 
 * conexões entre dois neurônios (origem e destino)
 * @example 
 * links_ws['0_to_1'] = 0.5
 * // 0 e 1 são os ids dos neurônios
 */
export let links_ws: {
    [link_name:string]: number
} = {}

/**
 * Limpa todos os pesos das conexões entre neurônios
 */
export function clearLinks() {
    links_ws = {}
}
export function setLinks(new_links: {[link_name:string]: number}) {
    links_ws = new_links
}
/**
 * unifica duas arrays, retornando uma nova array contendo todos os elementos das duas
 */
export const junct = <A, B>(arr1: A[], arr2:B[]):Array<[A,B, number]> => {
    return arr1.map((e, i) => [e, arr2[i], i])
}

/**
 * Escala/reduz um número entre dois valores
 */
export function scaleBetween(unscaledNum:number, minAllowed:number, maxAllowed:number, min:number, max:number) {
    return (maxAllowed - minAllowed) * (unscaledNum - min) / (max - min) + minAllowed;
}
/**
 * Retorna um valor aleatório entre min e max e o escala
 */
export function sigmoidRandom(min:number, max:number) {
    return scaleBetween(randomFromInterval(min, max), -1, 1, min, max)
}

/**
 * Retorna um valor aleatório entre min e max e o escala
 */
export function scale6(n:number) {
    return scaleBetween(n, -1, 1, 0, 6)
}

/**
 * Soma todos os valores de uma array de números
 */
export const sum = (arr:number[]) => arr.reduce((a, b) => a+b, 0)

/**
 * gera um número aleatório entre um intervalo (inclusivo)
 * @param min início do intervalo
 * @param max fim do intervalo
 * @returns número aleatório
 */
export function randomFromInterval(min:number, max:number) { 
    return Math.random() * (max - min) + min
}

/**
 * Cria uma nova conexão entre dois neurônios e retorna o peso da conexão
 * @param origin neuronio de origem
 * @param end neuronio de destino
 * @returns peso da conexão entre os dois neurônios
 */
function createLink(origin: Neuron, end: Neuron) {
    links_ws[origin.id+'_to_'+end.id] = randomFromInterval(-1, 1)
    return links_ws[origin.id+'_to_'+end.id]
}

/**
 * Retorna o peso da conexão entre dois neurônios
 * @param origin neuronio de origem
 * @param end neuronio de destino
 * @returns peso da conexão entre os dois neurônios
 */
function findLinkWeight(origin: Neuron, end: Neuron) {
    return links_ws[origin.id+'_to_'+end.id]
}
/**
 * Retorna o nome da conexão entre dois neurônios
 * @param origin neuronio de origem
 * @param end neuronio de destino
 * @returns nome da conexão entre os dois neurônios
 */
function findLink(origin: Neuron, end: Neuron) {
    return origin.id+'_to_'+end.id
}

/**
 * Resultado após aplicar a função de entrada sobre os inputs do neurônio e a função de ativação sobre o resultado da função de entrada
 */
type NeuronOutput = {
    /**
     * Neurônio que originou o valor
     */
    origin: Neuron;
    /**
     * Soma ponderada dos valores de entrada
     * `v_{i}(n)`
     */
    weighted_sum?: number;
    /**
     * Valor da saída do neurônio
     * `y_{i}(n)`
     */
    value: number;
}

// Armazena o valor do próximo ID a ser usado em um novo neurônio
let next_id = 0;

/**
 * Classe que representa um neurônio
 */
export class Neuron {
    id: number;
    constructor() {
        this.id = next_id;
        next_id++;
    }
    /**
     * Retorna um objeto contendo o neuronio 
     * e o valor da saída dele, ou, `y`
     */
    receive(inputs: NeuronOutput[]) {
        const weighted_sum = this.input_function(inputs)
        
        return {
            origin: this,
            weighted_sum,
            value: this.activation_function(weighted_sum)
        }
    }
    /**
     * Computa os pesos e retorna o valor que vai ser passado para a função de ativação
     */
    input_function(inputs: NeuronOutput[]): number {
        let sum = 0;
        for (const { value, origin } of inputs) {
            // Each unit j first computes a weighted sum of its inputs:
            sum += value * findLinkWeight(origin, this)
        }
        return sum
    }
    /**
     * Função de ativação sigmoid
     */
    φ = this.activation_function;
    /**
     * Derivada da função de ativação sigmoid
     */
    φ_derivative(input: number): number {
        return Math.exp(-input) / Math.pow(1 + Math.exp(-input), 2)
    }
    φ_derivative_alt(input: number): number {
        return this.φ(input) * (1 - this.φ(input))
    }
    /**
     * Função de ativação sigmoid
     */
    activation_function(input: number): number {
        return 1.0/(1.0 + Math.exp(-input))
    }
}

/**
 * Classe que representa neurônios de entrada
 */
class InputNeuron extends Neuron {
    in_value: number
    constructor() {
        super()
        this.in_value = 0;
    }
}

const activation_functions = {
    sigmoid: (input: number) => 1.0/(1.0 + Math.exp(-input)),
    sigmoid_derivative: (input: number) => Math.exp(-input) / Math.pow(1 + Math.exp(-input), 2),
    relu: (input: number) => input > 0 ? input : 0,
    relu_derivative: (input: number) => input > 0 ? 1 : 0,
    tanh: (input: number) => Math.tanh(input),
    tanh_derivative: (input: number) => 1 - Math.tanh(input) * Math.tanh(input),
}

/**
 * Configuração de uma camada, utilizada na criação de camadas dinamicamente
 */
export type LayerConfig = {
    inline_bias?: boolean
    /**
     * Função de ativação de todos os neuronios da camada
     */
    activation_function?: 'sigmoid' | 'relu' | 'tanh';
    /**
     * True se a camada for uma camada de entrada
     */
    is_input?: boolean,
    /**
     * True se a camada for uma camada de saída
     */
    is_output?: boolean,
    /**
     * Número de neurônios na camada
     */
    neurons_number: number,
    /**
     * Neurônio de bias
     */
    bias?: boolean,
}
/**
 * Configuração de uma camada, utilizada no processamento da rede
 */
interface LayerConfigInner extends LayerConfig {
    is_hidden: boolean
}
type Layer = {
    config: LayerConfigInner, 
    neurons: Neuron[]
};

export type TrainConfig = {
    /**
     * Número de vezes que o algoritmo de treinamento será executado
     */
    epochs: number,
    momentum?: number,
    debug?: boolean,
    silent?: boolean,
    /**
     * Número de vezes que o algoritmo de treinamento será executado por epoch
     */
    iteracoes: number,
    /**
     * Taxa de aprendizado da rede
     */
    taxa_aprendizado: number,
    /**
     * Funcão de utilizada ao fim de cada epoch para verificar se uma parada do treinamento deve ser feita
     */ 
    stop_condition?: (epoch: number, error: number) => boolean,
    on_epoch_end?: (epoch?: number, error?: number) => Promise<void>,
    training_set: {inputs: number[], desired_outputs: number[]}[],
}

type TrainIterationInfo = {
    iteration_index: number,
    last_Δw_global: {[key: string]: number},
}

const softmax = (vector:number[]) => {
    let sum = 0;
    for (let i = 0; i < vector.length; i++) {
        sum += Math.exp(vector[i]);
    }
    return vector.map(x => Math.exp(x) / sum)
}

/**
 * @note A rede só suporta 1 neuronio de bias na camada de entrada
 */
export class NeuralNetwork {
    layers: Layer[];
    constructor() {
        this.layers = [];
    }
    pushLayer(layer_config: LayerConfig) {
        // Verifica se a camada é oculta
        const is_hidden = !layer_config.is_input && !layer_config.is_output;
        const NeuronType = layer_config.is_input ? InputNeuron : Neuron
        // Cria a array de neurônios da camada
        const neurons:Neuron[] = [];

        // Adiciona o bias a camada de entrada caso necessário
        if (layer_config.bias) {
            const bias = new InputNeuron();
            bias.in_value = 1;
            neurons.push(bias);
        }
        for (let i = 0; i < layer_config.neurons_number; i++) {
            const neuron = new NeuronType()
            if (layer_config.activation_function) {
                neuron.activation_function = activation_functions[layer_config.activation_function]
                neuron.φ_derivative = activation_functions[`${layer_config.activation_function}_derivative`]
            }
            neurons.push(neuron)
        }
        
        if (layer_config.inline_bias) {
            for (const neuron of neurons) {
                neuron.input_function = (inputs: NeuronOutput[]): number => {
                    let sum = 0;
                    for (const { value, origin } of inputs) {
                        // Each unit j first computes a weighted sum of its inputs:
                        sum += value * findLinkWeight(origin, neuron)
                    }
                    return sum + (1 * findLinkWeight(neuron, neuron))
                }
            }
        }

        // Adiciona a cama a rede
        this.layers.push({
            config: {...layer_config, is_hidden},
            neurons: neurons
        })
    }
    /**
     * Cria os pesos de cada conexão entre os neurônios
     */
    createWeights() {
        for (const [index, layer] of this.layers.entries()) {
            if (layer.config.inline_bias) {
                for (const neuron of layer.neurons) {
                    createLink(neuron, neuron)
                }
            }
            for (const neuron of layer.neurons) {
                if (!this.layers[index+1]) return;
                for (const neuron_of_next_layer of this.layers[index+1].neurons) {
                    createLink(neuron, neuron_of_next_layer)
                }
            }
        }
    }
    train_iteration(inputs:number[], desired_outputs:number[], config: TrainConfig, iteration_info:TrainIterationInfo):number {
        const input_layer = this.layers[0];
        /**
         * Adiciona os valores de entrada aos neurônios de entrada
         */
        for (const [index, neuron] of (input_layer.neurons as InputNeuron[]).entries()) {
            neuron.in_value = inputs[index];
        }
        /**
         * Salva o valor de Y de todos os neuronios de todas as camadas em ordem sequencial 
         * (a camada de saida não é considerada) 
         * (a camada de entrada é o primeiro index da array)
         */
        const layer_neuron_outputs:NeuronOutput[][] = [
            (input_layer.neurons as InputNeuron[]).map(neuron => ({ origin: neuron, value: neuron.in_value }) )
        ];

        /**
         * Alimenta os valores para frente, sempre se baseando na última camada que foi alimentada
         */
        for (const { config, neurons } of this.layers) {
            if (config.is_output) {
                // o resultado da camada de saida é criado fora do loop para preservar o valor de Y nos calculos iniciais da retropropagação
                // ps.: talvez esse valor poderia ser armazenado na variavel layer_neuron_outputs também mas escolhi assim para acessar a variavel mais facilmente
                break;
            }
            if (config.is_hidden) {
                const hidden_response:NeuronOutput[] = []
                for (const neuron of neurons) {
                    // Alimentando o resultado da ultima camada
                    hidden_response.push(neuron.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
                }
                layer_neuron_outputs.push(hidden_response);
            }
        }
        //for (const [i,l] of layer_neuron_outputs.entries()) {
        //    console.log(i,l.map(n => n.value).join(', '))
        //}
        
        // Resultado da camada de saida
        const output_response:NeuronOutput[] = [];
        for (const unit of this.layers[this.layers.length - 1].neurons) {
            // Alimentando o resultado da ultima camada escondida antes da camada de saida
            output_response.push(unit.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
        }        

        /**
         * E = Erro global instantâneo (nessa iteração)
         * E(n) = 1/2  \sum_{j=1}^{J} e^2_{j}(n)
         */
        let E = (1/2)*sum(output_response.map((unit, j) => (desired_outputs[j] - unit.value)**2))
        
        // Retropropagação

        // armazenar todos os deltas no mesmo lugar pra atualizar todos os pesos no fim da iteração
        const Δw_global:Array<[number, string]> = []

        // armazenar todos os valores das gradientes de cada camada (de forma contrária ao fluxo de dados, a camada de saida vai estar no index 0)
        const δ_layers:{ origin: Neuron, value: number }[][] = []
        
        // Ref.: https://stackoverflow.com/questions/30610523/reverse-array-in-javascript-without-mutating-original-array
        for (const [layer_index, layer] of this.layers.slice().reverse().entries()) {
            const layer_neuron_output = layer_neuron_outputs[layer_neuron_outputs.length - layer_index];
            if (layer.config.is_output) {
                /**
                 * Calculo de gradiente local dos neuronios na camada de saida
                 * é bem simples já que a gente pode usar o calculo de erro com base na saida desejada
                 * δ_{j}(n)=-e_{j}(n)φ'_{j}(v_{j}(n))
                 * δ_{j}(n) = -(desired[j] - unit.value) * unit.φ'(unit.weighted_sum)
                 */
                const δ_saida:{ origin: Neuron, value: number }[] = []
                for (const [neuron, output, j]  of junct(layer.neurons, output_response)) {
                    const δ = -(desired_outputs[j] - output.value) * neuron.φ_derivative(output.weighted_sum as number)
                    δ_saida.push({ origin: neuron, value: δ })
                }
                δ_layers.push(δ_saida)
                /**
                 * Calculo de delta de pesos entre a camada oculta antes da camada de saida e a camada de saida
                 * Δw_{ij}=-ηδ_{j}(n)y_{i}(n)
                 */
                for (const output of layer_neuron_outputs[layer_neuron_outputs.length - 1]) {
                    for (const δ of δ_saida) {
                        let Δw = -config.taxa_aprendizado * δ.value * output.value;
                        if (config.momentum) {
                            Δw += (config.momentum * iteration_info.last_Δw_global[findLink(output.origin, δ.origin)] || 0)
                        }
                        Δw_global.push([Δw, findLink(output.origin, δ.origin)])
                    }
                }
            } else if (layer.config.is_hidden) {
                /**
                 * Calculo de gradiente local dos neuronios na camada escondida (J)
                 * δ_{j}(n)=φ'_{j}(v_{j}(n)) \sum_{i=1}^{I}δ_{i}(n)w_{ji}
                 */
                const δ_escondida:{ origin: Neuron, value: number }[] = []
                //console.log(layer.config.id, ff_layers[layer_index - 1].config.id)
                for (const [neuron, output, j] of junct(layer.neurons, layer_neuron_output)) {
                    const δ = neuron.φ_derivative(output.weighted_sum as number) * sum(δ_layers[layer_index - 1].map(({value: grad_local, origin}, j) => grad_local * findLinkWeight(output.origin, origin)))
                    δ_escondida.push({ origin: neuron, value: δ })
                }
                δ_layers.push(δ_escondida)
                
                /**
                 * Calculo de delta de pesos entre a camada (J-1) e a camada oculta (J)
                 * Δw_{ij}=-ηδ_{j}(n)y_{i}(n)=ηδ_{j}(n)x_{i}(n)
                 */
                for (const output of layer_neuron_outputs[layer_neuron_outputs.length - 1 - layer_index]) {
                    for (const δ of δ_escondida) {
                        let Δw = -config.taxa_aprendizado * δ.value * output.value;
                        if (config.momentum) {
                            Δw += (config.momentum * iteration_info.last_Δw_global[findLink(output.origin, δ.origin)] || 0)
                        }
                        Δw_global.push([Δw, findLink(output.origin, δ.origin)])
                    }
                }
            } else if (layer.config.is_input) {
                // não faz nada
            }
        }
        //console.log(Δw_global)
        for (const [ Δw, link_name ] of Δw_global) {
            links_ws[link_name]+= Δw
            if (config.momentum) {
                iteration_info.last_Δw_global[link_name] = Δw
            }
        }

        return E;
    }
    /**
     * Imita a propagação de uma entrada da rede até a camada de saida, retornando o resultado da saida
     * @param _inputs array de arrays de valores de entrada
     * @returns O chute da rede neural
     */
    guess(_inputs:number[]) {
        let inputs:number[] = [..._inputs];
        const input_layer = this.layers[0];
        if (input_layer.config.bias) {
            inputs.unshift(1);
        }
        for (const [i, neuron] of (input_layer.neurons as InputNeuron[]).entries()) {
            neuron.in_value = inputs[i];
        }
        const layer_neuron_outputs:NeuronOutput[][] = [(input_layer.neurons as InputNeuron[]).map(neuron => ({ origin: neuron, value: neuron.in_value }) )];
        
        for (const { config, neurons } of this.layers) {
            if (config.is_output) {
                break;
            }
            if (config.is_hidden) {
                const hidden_response:NeuronOutput[] = []
                for (const neuron of neurons) {
                    // Alimentando o resultado da ultima camada
                    hidden_response.push(neuron.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
                }
                layer_neuron_outputs.push(hidden_response);
            }
        }
        
        // Resultado da camada de saida
        const output_response:NeuronOutput[] = [];
        for (const unit of this.layers[this.layers.length - 1].neurons) {
            // Alimentando o resultado da ultima camada escondida antes da camada de saida
            output_response.push(unit.receive(layer_neuron_outputs[layer_neuron_outputs.length - 1]))
        }
        return {output:output_response.map(e=>e.value), layer_neuron_outputs, output_response};
    }
    test(test_set:{inputs: number[], desired_outputs: number[]}[], test_iteration?: (err: number, output: number[], desired: number[]) => any) {
        for (const {inputs, desired_outputs} of test_set) {
            const {output} = this.guess(inputs);

            const error = sum(desired_outputs.map((desired, i) => Math.abs(desired - output[i])))
            test_iteration && test_iteration(error, output, desired_outputs)
        }
    }
    async train(config: TrainConfig) {
        const Ēs:number[] = []
        let last_Δw_global = {}
        for await (const epoch of new Array(config.epochs).fill(0).map((_, i) => i)) {
            let error = 0;
            // wait for 100ms
            await new Promise(resolve => setTimeout(resolve, 100));

            for (let i = 0; i < config.iteracoes; i++) {
                // Pega um conjunto de treinamento aleatório
                const training_item = config.training_set[Math.floor(Math.random() * config.training_set.length)]
                // Executa o treinamento
                let inputs = training_item.inputs;
                if (this.layers[0].config.bias) {
                    inputs = [1, ...inputs]
                }
                if (config.debug && i % 100 == 0) {
                    console.log(`Epoch: ${epoch}/${config.epochs} Iteração: ${i}/${config.iteracoes}`)
                }
                error += this.train_iteration(inputs, training_item.desired_outputs, config, { iteration_index: i, last_Δw_global })
            }
            const Ē = (1/config.iteracoes)*error
            if (!config.silent) {
                console.log({epoch, erro_medio: Ē})
            }
            Ēs.push(Ē)
            if (config.on_epoch_end) {
                await config.on_epoch_end(epoch, Ē)
            }
            // Verifica se a parada deve ser feita
            if (config.stop_condition && config.stop_condition(epoch, error)) break;
        }

        
        const train_result = {
            epochs: config.epochs,
            mean_error: sum(Ēs)/Ēs.length,
            std_error: Math.sqrt(sum((Ēs.map(Ē => (Ē - sum(Ēs)/Ēs.length)**2)))/Ēs.length),
            min_error: Math.min(...Ēs),
            max_error: Math.max(...Ēs),
            error_diff: Math.max(...Ēs) - Math.min(...Ēs),
            last_error: Ēs[Ēs.length - 1],
            taxa_aprendizado: config.taxa_aprendizado,
            iteracoes_por_epoch: config.iteracoes,
        }
        if (!config.silent) {
            console.log(train_result)
        }
        return train_result
    }
}