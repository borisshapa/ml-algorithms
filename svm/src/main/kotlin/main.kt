import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

private fun readLn() = readLine()!!
private fun readInt() = readLn().toInt()
private fun readLong() = readLn().toLong()
private fun readDouble() = readLn().toDouble()
private fun readStrings() = readLn().split(" ")
private fun readInts() = readStrings().map { it.toInt() }
private fun readLongs() = readStrings().map { it.toLong() }
private fun readDoubles() = readStrings().map { it.toDouble() }

operator fun List<Double>.plus(v: List<Double>): List<Double> =
    this.zip(v).map { it.first + it.second }

operator fun List<Double>.minus(v: List<Double>): List<Double> =
    this.zip(v).map { it.first - it.second }

operator fun List<Double>.times(x: Double): List<Double> =
    this.map { it * x }

operator fun Double.times(v: List<Double>): List<Double> =
    v * this

fun List<Double>.dot(v: List<Double>): Double =
    this.zip(v).map { it.first * it.second }.reduce { a, b -> a + b }

fun List<List<Double>>.dot(mat: List<List<Double>>): List<List<Double>> {
    val n = this.size
    val m = mat[0].size
    val transpose = List(mat[0].size) { i -> List(mat.size) { j -> mat[j][i] } }
    return List(n) { i -> List(m) { j -> this[i].dot(transpose[j]) } }
}

class SVM(val epochs: Int = 10, val c: Double = 1.0) {
    var k: List<List<Double>> = emptyList()
    var y: List<Double> = emptyList()
    var alphas: MutableList<Double> = mutableListOf()
    var bias = 0.0

    fun predict(ind: Int): Double {
        val kernels = List(k.size) { i -> k[i][ind] }
        val pred = alphas.zip(y).map { it.first * it.second }.dot(kernels) + bias
        return pred
    }


    fun error(ind: Int): Double {
        val yi = y[ind]
        return predict(ind) - yi
    }

    fun fit(kToFit: List<List<Double>>, yToFit: List<Double>) {
        k = kToFit
        y = yToFit
        alphas = MutableList(k.size) { 0.0 }

        for (epoch in 0 until epochs) {
            val objInd1 = Random.nextInt(k.size)
            var objInd2 = objInd1
            while (objInd2 == objInd1) {
                objInd2 = Random.nextInt(k.size)
            }

            val eta = 2.0 * k[objInd1][objInd2] - (k[objInd1][objInd1] + k[objInd2][objInd2])
            if (eta >= 0.0) {
                continue
            }

            var lower = 0.0
            var upper = 0.0
            if (y[objInd1] == y[objInd2]) {
                lower = max(0.0, alphas[objInd1] + alphas[objInd2] - c)
                upper = min(c, alphas[objInd1] + alphas[objInd2])
            } else {
                lower = max(0.0, alphas[objInd1] - alphas[objInd2])
                upper = min(c, c + alphas[objInd1] - alphas[objInd2])
            }

            val objError2 = error(objInd2)
            val objError1 = error(objInd1)

            val oldAlpha1 = alphas[objInd1]
            val oldAlpha2 = alphas[objInd2]

            alphas[objInd1] -= (y[objInd1] * (objError2 - objError1)) / eta
            alphas[objInd1] = min(alphas[objInd1], upper)
            alphas[objInd1] = max(alphas[objInd1], lower)
            alphas[objInd2] += y[objInd2] * y[objInd1] * (oldAlpha1 - alphas[objInd1])

            val bias2 = bias - objError2 - y[objInd2] * (alphas[objInd2] - oldAlpha2) * k[objInd2][objInd2]
            -y[objInd1] * (alphas[objInd1] - oldAlpha1) * k[objInd1][objInd2]
            val bias1 = bias - objError1 - y[objInd1] * (alphas[objInd1] - oldAlpha1) * k[objInd1][objInd1]
            -y[objInd2] * (alphas[objInd2] - oldAlpha2) * k[objInd1][objInd2]

            bias = when {
                0.0 < alphas[objInd2] && alphas[objInd2] < c -> bias2
                0.0 < alphas[objInd1] && alphas[objInd1] < c -> bias1
                else -> (bias2 + bias1) / 2.0
            }
        }
    }
}

fun getKY(objects: List<List<Double>>): Pair<List<List<Double>>, List<Double>> =
    Pair(objects.map { it.dropLast(1) }, objects.map { it.last() })


fun main() {
    val n = readInt()
    val objects = List(n) { readDoubles() }
    val (k, y) = getKY(objects)
    val c = readDouble()
    val svm = SVM(60000, c)
    svm.fit(k, y)

    svm.alphas.forEach { println(it) }
    println(svm.bias)
}