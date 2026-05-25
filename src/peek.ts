import tf, { loadLayersModel } from "@tensorflow/tfjs-node"

async function loadModel(modelPath: string) {
    await tf.ready()
    return loadLayersModel(modelPath)
}

function precision1D(row: number[], dp: number) {
    return row.map(r => parseFloat(r.toFixed(dp)))
}

function precision2D(matrix: number[][], dp: number) {
    let output: number[][] = []
    for (let i=0; i<matrix.length; i++) {
        const row = matrix[i]
        const precRow: number[] = row.map(r => parseFloat(r.toFixed(dp)))
        output.push(precRow)
    }
    return output
}

export type WeightStats = {
    min: number,
    max: number,
    mean: number,
    std: number,
    sparsity: number,
}

function computeStats(values: number[], dp: number): WeightStats {
    const n = values.length
    const min = Math.min(...values)
    const max = Math.max(...values)
    const mean = values.reduce((s, v) => s + v, 0) / n
    const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / n)
    const sparsity = values.filter(v => Math.abs(v) < 0.01).length / n
    return {
        min: parseFloat(min.toFixed(dp)),
        max: parseFloat(max.toFixed(dp)),
        mean: parseFloat(mean.toFixed(dp)),
        std: parseFloat(std.toFixed(dp)),
        sparsity: parseFloat(sparsity.toFixed(dp)),
    }
}

export type LayerData = {
    name: string,
    config: tf.serialization.ConfigDict,
    weights: { shape: number[], values: number[][], stats: WeightStats },
    bias: { shape: number[], values: number[], stats: WeightStats },
}

function extractLayers(model: tf.LayersModel): LayerData[] {
    const layers: LayerData[] = []

    model.layers.forEach((layer) => {
        const config = layer.getConfig()

        if (layer.weights.length === 0) return

        const weightsTensor = layer.weights[0].read()
        const biasesTensor = layer.weights[1].read()
        const weightsValues = weightsTensor.arraySync() as number[][]
        const biasValues = biasesTensor.arraySync() as number[]

        layers.push({
            name: layer.name,
            config,
            weights: {
                shape: weightsTensor.shape,
                values: precision2D(weightsValues, 2),
                stats: computeStats(weightsValues.flat(), 2),
            },
            bias: {
                shape: biasesTensor.shape,
                values: precision1D(biasValues, 2),
                stats: computeStats(biasValues, 2),
            },
        })
    })
    return layers
}

function printLayers(layers: LayerData[]) {
    console.log(JSON.stringify(layers, null, 2))
}

export async function peekLayers(modelPath: string): Promise<LayerData[]> {
    const model = await loadModel(modelPath)
    return extractLayers(model)
}

export async function toStdOut(modelPath: string) {
    const layers = await peekLayers(modelPath)
    printLayers(layers)
}
