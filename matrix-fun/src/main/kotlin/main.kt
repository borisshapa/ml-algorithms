import kotlin.math.tanh

private fun readLn() = readLine()!!
private fun readInt() = readLn().toInt()
private fun readLong() = readLn().toLong()
private fun readDouble() = readLn().toDouble()
private fun readStrings() = readLn().split(" ")
private fun readInts() = readStrings().map { it.toInt() }
private fun readLongs() = readStrings().map { it.toLong() }
private fun readDoubles() = readStrings().map { it.toDouble() }

typealias Matrix = Array<DoubleArray>

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

abstract class Node {
    lateinit var result: Matrix

    protected abstract fun eval()
    abstract fun backProp()

    fun evaluate() {
        eval()
        if (!this::diff.isInitialized) {
            diff = Array(result.size) { DoubleArray(result.first().size) }
        }
    }

    lateinit var diff: Matrix
}

class Var(val r: Int, val c: Int) : Node() {
    fun setMatrix(matrix: Array<DoubleArray>) {
        result = matrix
    }

    override fun eval() {
    }

    override fun backProp() {
    }
}

class Tnh(private val node: Node) : Node() {
    override fun eval() {
        result = node.result.copy()
        for (i in result.indices) {
            for (j in result[i].indices) {
                result[i][j] = tanh(result[i][j])
            }
        }
    }

    override fun backProp() {
        for (i in result.indices) {
            for (j in result[i].indices) {
                node.diff[i][j] += (1 - result[i][j] * result[i][j]) * diff[i][j]
            }
        }
    }
}

class Rlu(private val alpha: Double, private val node: Node) : Node() {
    override fun eval() {
        result = node.result.copy()
        for (i in result.indices) {
            for (j in result[i].indices) {
                if (result[i][j] < 0) {
                    result[i][j] *= alpha
                }
            }
        }
    }

    override fun backProp() {
        for (i in result.indices) {
            for (j in result[i].indices) {
                node.diff[i][j] += (if (node.result[i][j] < 0) alpha else 1.0) * diff[i][j]
            }
        }
    }
}

class Mul(private val node1: Node, private val node2: Node) : Node() {
    override fun eval() {
        result = node1.result.dot(node2.result)
    }

    override fun backProp() {
        val da = diff.dot(node2.result.T)
        val db = node1.result.T.dot(diff)

        node1.diff += da
        node2.diff += db
    }
}

class Sum(private val nodes: List<Node>) : Node() {
    override fun eval() {
        result = nodes.first().result.copy()
        for (i in 1..nodes.lastIndex) {
            result += nodes[i].result
        }
    }

    override fun backProp() {
        for (node in nodes) {
            node.diff += diff
        }
    }
}

class Had(private val nodes: List<Node>) : Node() {
    override fun eval() {
        result = nodes.first().result.copy()
        for (i in 1..nodes.lastIndex) {
            result = result.matMul(nodes[i].result)
        }
    }

    override fun backProp() {
        for (i in result.indices) {
            for (j in result[i].indices) {
                for ((node1Ind, node1) in nodes.withIndex()) {
                    var product = diff[i][j]
                    for ((node2Ind, node2) in nodes.withIndex()) {
                        if (node1Ind != node2Ind) {
                            product *= node2.result[i][j]
                        }
                    }
                    node1.diff[i][j] += product
                }
            }
        }
    }
}


fun main() {
    val (n, m, k) = readInts()
    val nodes = ArrayList<Node>(n)
    for (i in 0 until n) {
        val input = readStrings()
        val type = input.first()
        val params = input.drop(1).map { it.toInt() }
        var node: Node? = null
        when (type) {
            "var" -> node = Var(params[0], params[1])
            "tnh" -> node = Tnh(nodes[params[0] - 1])
            "rlu" -> node = Rlu(1.0 / params[0], nodes[params[1] - 1])
            "mul" -> node = Mul(nodes[params[0] - 1], nodes[params[1] - 1])
            "sum" -> node = Sum(params.drop(1).map { nodes[it - 1] })
            "had" -> node = Had(params.drop(1).map { nodes[it - 1] })
        }
        if (node != null) {
            nodes.add(node)
        }
    }

    for (i in 0 until m) {
        val node = nodes[i] as Var
        val matrix = Array(node.r) { doubleArrayOf() }
        for (j in 0 until node.r) {
            matrix[j] = readDoubles().toDoubleArray()
        }
        node.setMatrix(matrix)
    }
    for (node in nodes) {
        node.evaluate()
    }
    for (i in 0 until k) {
        for (row in nodes[n - k + i].result) {
            row.forEach { print("$it ") }
            println()
        }
    }

    for (i in 0 until k) {
        val node = nodes[n - k + i]
        val matSize = node.diff.size
        val diff = Array(matSize) { doubleArrayOf() }
        for (j in node.diff.indices) {
            diff[j] = readDoubles().toDoubleArray()
        }
        node.diff = diff
    }

    for (i in nodes.size - 1 downTo 0) {
        nodes[i].backProp()
    }
    for (i in 0 until m) {
        for (row in nodes[i].diff) {
            row.forEach { print("$it ") }
            println()
        }
    }
}