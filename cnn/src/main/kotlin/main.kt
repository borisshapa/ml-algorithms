import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

private fun readLn() = readLine()!!
private fun readInt() = readLn().toInt()
private fun readLong() = readLn().toLong()
private fun readDouble() = readLn().toDouble()
private fun readStrings() = readLn().split(" ")
private fun readInts() = readStrings().map { it.toInt() }
private fun readLongs() = readStrings().map { it.toLong() }
private fun readDoubles() = readStrings().map { it.toDouble() }

typealias Matrix = Array<DoubleArray>
typealias Tensor = Array<Matrix>

fun createTensor(n: Int, m: Int, k: Int): Tensor {
    return Array(n) {
        Array(m) {
            DoubleArray(k)
        }
    }
}

fun Tensor.copy() = map { it.copy() }.toTypedArray()
fun Matrix.copy() = map { it.clone() }.toTypedArray()

fun Matrix.dot(mat: Matrix): Matrix {
    val result = Array(this.size) { DoubleArray(mat.first().size) }
    for (i in result.indices) {
        for (j in result[i].indices) {
            result[i][j] = this[i].zip(mat.map { it[j] }).sumByDouble { it.first * it.second }
        }
    }
    return result
}

fun Matrix.matMul(mat: Matrix): Matrix {
    val result = this.copy()
    for (i in result.indices) {
        for (j in result[i].indices) {
            result[i][j] *= mat[i][j]
        }
    }
    return result
}

operator fun Matrix.plus(mat: Matrix): Matrix {
    val result = this.copy()
    for (i in result.indices) {
        for (j in result[i].indices) {
            result[i][j] += mat[i][j]
        }
    }
    return result
}

val Matrix.T: Matrix
    get() = Array(this.first().size) { i -> DoubleArray(this.size) { j -> this[j][i] } }

fun Tensor.applyToEveryElement(f: (Double) -> Double) {
    for (i in this.indices) {
        for (j in this[i].indices) {
            for (k in this[i][j].indices) {
                this[i][j][k] = f(this[i][j][k])
            }
        }
    }
}

abstract class Node {
    lateinit var result: Tensor

    protected abstract fun eval()
    abstract fun backProp()

    lateinit var diff: Tensor

    fun evaluate() {
        eval()
        if (!this::diff.isInitialized) {
            diff = createTensor(result.size, result.first().size, result.first().first().size)
        }
    }
}

class Var : Node() {
    fun setTensor(tensor: Tensor) {
        result = tensor
    }

    override fun eval() {
    }

    override fun backProp() {
    }

}

class Relu(private val alpha: Double, private val node: Node) : Node() {
    override fun eval() {
        result = node.result.copy()
        result.applyToEveryElement {
            if (it < 0) it * alpha else it
        }
    }

    override fun backProp() {
        for (i in diff.indices) {
            for (j in diff[i].indices) {
                for (k in diff[i][j].indices) {
                    node.diff[i][j][k] += (if (node.result[i][j][k] < 0) alpha else 1.0) * diff[i][j][k]
                }
            }
        }
    }
}

class Pool(private val s: Int, private val node: Node) : Node() {
    lateinit var maxIndices: Array<Array<Array<List<Pair<Int, Int>>>>>

    override fun eval() {
        val nodeResult = node.result
        result = Array((nodeResult.size + s - 1) / s) {
            Array((nodeResult.first().size + s - 1) / s) {
                DoubleArray(nodeResult.first().first().size) { Double.NEGATIVE_INFINITY }
            }
        }
        maxIndices = Array(result.size) {
            Array(result.first().size) {
                Array(result.first().first().size) {
                    arrayListOf()
                }
            }
        }

        for (i in result.indices) {
            for (j in result[i].indices) {
                for (k in result[i][j].indices) {
                    var max = arrayListOf<Pair<Int, Int>>()
                    for (windowI in 0 until s) {
                        for (windowJ in 0 until s) {
                            val valueI = i * s + windowI
                            val valueJ = j * s + windowJ
                            if (valueI < node.result.size && valueJ < node.result.first().size) {
                                val value = node.result[valueI][valueJ][k]
                                if (value > result[i][j][k]) {
                                    result[i][j][k] = value
                                    max = arrayListOf(Pair(valueI, valueJ))
                                }
                                if (value == result[i][j][k]) {
                                    max.add(Pair(valueI, valueJ))
                                }
                            }
                        }
                    }
                    maxIndices[i][j][k] = max
                }
            }
        }
    }

    override fun backProp() {
        for (i in diff.indices) {
            for (j in diff[i].indices) {
                for (k in diff[i][j].indices) {
                    for ((maxI, maxJ) in maxIndices[i][j][k]) {
                        node.diff[maxI][maxJ][k] = diff[i][j][k]
                    }
                }
            }
        }
    }
}

class Bias(private val b: List<Double>, private val node: Node) : Node() {
    lateinit var diffByParameters: Array<Double>

    override fun eval() {
        result = node.result.copy()
        for (i in result.indices) {
            for (j in result[i].indices) {
                for (k in result[i][j].indices) {
                    result[i][j][k] += b[k]
                }
            }
        }
    }

    override fun backProp() {
//        printTensor(node.diff)
//        printTensor(diff)
//        println(diff.first().first().size)
        for (i in diff.indices) {
            for (j in diff[i].indices) {
                for (k in diff[i][j].indices) {
                    node.diff[i][j][k] += diff[i][j][k]
                }
            }
        }
        
        diffByParameters = Array(result.first().first().size) { 0.0 }
        for (i in diff.indices) {
            for (j in diff.first().indices) {
                for (k in diff.first().first().indices) {
                    diffByParameters[k] += diff[i][j][k]
                }
            }
        }
    }
}

abstract class Cnv(
    protected val s: Int,
    protected val p: Int,
    protected val kernel: Array<Tensor>,
    protected val node: Node
) : Node() {
    lateinit var diffByParameters: Array<Tensor>
    lateinit var valIJ: Array<Array<Pair<Int, Int>>>
    lateinit var expandTensor: Tensor

    abstract fun expand(tensor: Tensor)

    override fun eval() {
        val kernelSize = kernel.first().first().size
        result = createTensor(
            (node.result.size + 2 * p - kernelSize) / s + 1,
            (node.result.first().size + 2 * p - kernelSize) / s + 1,
            kernel.size
        )

        expand(node.result)

        for (i in result.indices) {
            for (j in result[i].indices) {
                for (k in result[i][j].indices) {
                    var conv = 0.0
                    for (windowI in 0 until kernelSize) {
                        for (windowJ in 0 until kernelSize) {
                            for (dim in node.result.first().first().indices) {
                                conv += expandTensor[i * s + windowI][j * s + windowJ][dim] *
                                        kernel[k][dim][windowI][windowJ]
                            }
                        }
                    }
                    result[i][j][k] = conv
                }
            }
        }
    }

    override fun backProp() {
        for (i in diff.indices) {
            for (j in diff[i].indices) {
                for (k in diff[i][j].indices) {

                    val kernelSize = kernel.first().first().size
                    for (windowI in 0 until kernelSize) {
                        for (windowJ in 0 until kernelSize) {
                            for (dim in node.result.first().first().indices) {
                                val (valI, valJ) = valIJ[i * s + windowI][j * s + windowJ]
                                node.diff[valI][valJ][dim] += diff[i][j][k] *
                                        kernel[k][dim][windowI][windowJ]
                            }
                        }
                    }
                }
            }
        }

        val kernelN = kernel.size
        val kernelM = kernel.first().size
        val kernelK = kernel.first().first().size

        diffByParameters = Array(kernelN) {
            createTensor(kernelM, kernelK, kernelK)
        }

        for (i in diff.indices) {
            for (j in diff[i].indices) {
                for (k in diff[i][j].indices) {

                    val kernelSize = kernel.first().first().size
                    for (windowI in 0 until kernelSize) {
                        for (windowJ in 0 until kernelSize) {
                            for (dim in node.result.first().first().indices) {
                                diffByParameters[k][dim][windowI][windowJ] +=
                                    expandTensor[i * s + windowI][j * s + windowJ][dim] * diff[i][j][k]
                            }
                        }
                    }
                }
            }
        }
    }
}

class Cnvm(s: Int, p: Int, kernel: Array<Tensor>, node: Node) : Cnv(s, p, kernel, node) {
    override fun expand(tensor: Tensor) {
        val n = tensor.size
        val m = tensor.first().size

        val newN = n + 2 * p
        val newM = m + 2 * p
        expandTensor = createTensor(newN, newM, tensor.first().first().size)
        valIJ = Array(newN) { Array(newM) { Pair(0, 0) } }

        for (i in expandTensor.indices) {
            for (j in expandTensor[i].indices) {
                for (k in expandTensor[i][j].indices) {
                    var valI = abs(i - p)
                    var valJ = abs(j - p)

                    if (valI >= n) {
                        valI = 2 * n - valI - 2
                    }
                    if (valJ >= m) {
                        valJ = 2 * m - valJ - 2
                    }
                    expandTensor[i][j][k] = tensor[valI][valJ][k]
                    valIJ[i][j] = Pair(valI, valJ)
                }
            }
        }
    }
}

class Cnve(s: Int, p: Int, kernel: Array<Tensor>, node: Node) : Cnv(s, p, kernel, node) {
    override fun expand(tensor: Tensor) {
        val n = tensor.size
        val m = tensor.first().size

        val newN = n + 2 * p
        val newM = m + 2 * p
        expandTensor = createTensor(newN, newM, tensor.first().first().size)
        valIJ = Array(newN) { Array(newM) { Pair(0, 0) } }

        for (i in expandTensor.indices) {
            for (j in expandTensor[i].indices) {
                for (k in expandTensor[i][j].indices) {
                    val valI = min(max(i - p, 0), n - 1)
                    val valJ = min(max(j - p, 0), m - 1)
                    expandTensor[i][j][k] = tensor[valI][valJ][k]
                    valIJ[i][j] = Pair(valI, valJ)
                }
            }
        }
    }
}

class Cnvc(s: Int, p: Int, kernel: Array<Tensor>, node: Node) : Cnv(s, p, kernel, node) {
    override fun expand(tensor: Tensor) {
        val n = tensor.size
        val m = tensor.first().size

        val newN = n + 2 * p
        val newM = m + 2 * p
        expandTensor = createTensor(newN, newM, tensor.first().first().size)
        valIJ = Array(newN) { Array(newM) { Pair(0, 0) } }

        for (i in expandTensor.indices) {
            for (j in expandTensor[i].indices) {
                for (k in expandTensor[i][j].indices) {
                    val valI = ((i - p) + n * 2) % n
                    val valJ = ((j - p) + m * 2) % m
                    expandTensor[i][j][k] = tensor[valI][valJ][k]
                    valIJ[i][j] = Pair(valI, valJ)
                }
            }
        }
    }
}

fun getKernel(d: Int, params: List<Int>) : Triple<Int, Int, Array<Tensor>> {
    val h = params[0]
    val k = params[1]
    val s = params[2]
    val p = params[3]
    val convValues = params.drop(4)
    val conv = Array(h) { createTensor(d, k, k) }

    for (ii in 0 until h) {
        for (jj in 0 until d) {
            for (kk in 0 until k) {
                for (tt in 0 until k) {
                    conv[ii][jj][kk][tt] = convValues[d * k * k * ii + k * k * jj + k * kk + tt].toDouble()
                }
            }
        }
    }

    return Triple(s, p, conv)
}

fun printTensor(t: Tensor) {
    for (k in t.first().first().indices) {
        for (i in t.indices) {
            for (j in t[i].indices) {
                print("${t[i][j][k]} ")
            }
        }
    }
    println()
}

fun main() {
    val input = readInts()
    val n = input[0]
    val d = input[1]

    val values = input.drop(2)
    val t = createTensor(d, n, n)

    for (i in 0 until d) {
        for (j in 0 until n) {
            for (k in 0 until n) {
                t[i][j][k] = values[n * n * i + n * j + k].toDouble()
            }
        }
    }

    val tensor = createTensor(n, n, d)
    for (i in 0 until n) {
        for (j in 0 until  n) {
            for (k in 0 until d) {
                tensor[i][j][k] = t[k][i][j]
            }
        }
    }

    val nodes = arrayListOf<Node>(Var())
    val firstNode = (nodes.first() as Var)
    firstNode.setTensor(tensor)
    firstNode.evaluate()

    val nLayers = readInt()
    for (i in 0 until nLayers) {
        val inputLayer = readStrings()
        val type = inputLayer.first()
        val params = inputLayer.drop(1).map { it.toInt() }
        var node: Node? = null
        when (type) {
            "relu" -> node = Relu(1.0 / params[0], nodes[i])
            "pool" -> node = Pool(params[0], nodes[i])
            "bias" -> node = Bias(params.map { it.toDouble() }, nodes[i])
            "cnvm" -> {
                val dd = nodes[i].result.first().first().size
                val (s, p, kernel) = getKernel(dd, params)
                node = Cnvm(s, p, kernel, nodes[i])
            }
            "cnve" -> {
                val dd = nodes[i].result.first().first().size
                val (s, p, kernel) = getKernel(dd, params)
                node = Cnve(s, p, kernel, nodes[i])
            }
            "cnvc" -> {
                val dd = nodes[i].result.first().first().size
                val (s, p, kernel) = getKernel(dd, params)
                node = Cnvc(s, p, kernel, nodes[i])
            }
        }
        if (node != null) {
            nodes.add(node)
            node.evaluate()
        }
    }


    val diffValues = readInts()
    val lastNode = nodes[nodes.size - 1]
    val lastNodeDiff = lastNode.diff

    val diffN = lastNodeDiff.size
    val diffM = lastNodeDiff.first().size
    val diffK = lastNodeDiff.first().first().size

    val diffTen = createTensor(diffK, diffN, diffM)
    for (i in 0 until diffK) {
        for (j in 0 until diffN) {
            for (k in 0 until diffM) {
                diffTen[i][j][k] = diffValues[diffN * diffM * i + diffM * j + k].toDouble()
            }
        }
    }

    val diffTensor = createTensor(diffN, diffM, diffK)
    for (i in 0 until diffN) {
        for (j in 0 until diffM) {
            for (k in 0 until diffK) {
                diffTensor[i][j][k] = diffTen[k][i][j]
            }
        }
    }
    lastNode.diff = diffTensor

    for (i in nodes.size - 1 downTo 0) {
        nodes[i].backProp()
    }

    printTensor(nodes[nodes.size - 1].result)

    printTensor(nodes.first().diff)

    for (node in nodes) {
        if (node is Bias) {
            node.diffByParameters.forEach { print("$it ") }
            println()
        }
        if (node is Cnv) {
            val tens = node.diffByParameters
            for (i in tens) {
                for (j in i) {
                    for (k in j) {
                        for (tt in k) {
                            print("$tt ")
                        }
                    }
                }
            }
            println()
        }
    }
}