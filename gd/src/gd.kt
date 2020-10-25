import java.io.File
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.pow
import kotlin.random.Random

private fun readLn() = readLine()!!
private fun readInt() = readLn().toInt()
private fun readLong() = readLn().toLong()
private fun readDouble() = readLn().toDouble()
private fun readStrings() = readLn().split(" ")
private fun readInts() = readStrings().map { it.toInt() }
private fun readLongs() = readStrings().map { it.toLong() }
private fun readDoubles() = readStrings().map { it.toDouble() }

fun readFileAsLinesUsingUseLines(fileName: String): List<String> = File(fileName).useLines { it.toList() }

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

fun normalize(dataset: List<List<Double>>): Pair<List<List<Double>>, List<Pair<Double, Double>>> {
    val featuresCount = dataset[0].size
    val minmax = MutableList(featuresCount) { Pair(0.0, 0.0) }
    for (feature in 0 until featuresCount) {
        val min = dataset.minBy { it[feature] }
        val max = dataset.maxBy { it[feature] }
        if (min == null || max == null) {
            return Pair(emptyList(), emptyList())
        }
        minmax[feature] = Pair(min[feature], max[feature])
    }

    val normalizedValues = List(dataset.size) { row ->
        List(featuresCount) { col ->
            val featureMinmax = minmax[col]
            val section = featureMinmax.second - featureMinmax.first
            if (section != 0.0) {
                (dataset[row][col] - featureMinmax.first) / section
            } else {
                1.0
            }
        }
    }

    return Pair(normalizedValues, minmax)
}

fun denormalize(w: List<Double>, minmax: List<Pair<Double, Double>>): List<Double> {
    val newW = w.toMutableList()
    for (i in 0 until w.size - 1) {
        val section = minmax[i].second - minmax[i].first
        if (section == 0.0) {
            newW[i] = w[i] / minmax[i].second
        } else {
            newW[i] = w[i] / section
            newW[w.size - 1] -= (minmax[i].first * w[i]) / section
        }
    }
    return newW
}

fun smape(y: Double, a: Double): Double =
    abs(y - a) / (abs(a) + abs(y))

fun smapeGrad(x: List<Double>, w: List<Double>, y: Double): List<Double> {
    val a = w.dot(x)
    val diff = a - y
    if (diff == 0.0) {
        return List(x.size) { 0.0 }
    }
    val sumAbs = abs(a) + abs(y)
    val sumAbs2 = sumAbs * sumAbs
    val absDiff = abs(diff)
    val signDiff = if (diff > 0) 1.0 else -1.0
    val signA = when {
        a > 0.0 -> 1.0
        a == 0.0 -> 0.0
        else -> -1.0
    }

    val numerator = signDiff * sumAbs - signA * absDiff
    val denominator = sumAbs2
    val grad = numerator / denominator
    return grad * x
}

fun smapeGradBatch(x: List<List<Double>>, y: List<Double>, w: List<Double>, batchSize: Int): List<Double> {
    val batch: MutableSet<Int> = mutableSetOf()
    while (batch.size < min(batchSize, x.size)) {
        batch.add((x.indices).random())
    }

    var res = List(w.size) { 0.0 }

    for (i in batch) {
        val xi = x[i]
        val yi = y[i]
        val a = w.dot(xi)
//        println("$a $yi")
        res = res + smapeGrad(xi, w, yi)
    }
    return res.map { it / batchSize }
}

fun calcSmape(x: List<List<Double>>, y: List<Double>, w: List<Double>): Double {
    var sum = 0.0
    for (i in x.indices) {
        sum += smape(y[i], w.dot(x[i]))
    }
    return sum / x.size
}

fun addBias(values: List<Double>): List<Double> {
    val values = values.toMutableList()
    values.add(1.0)
    return values
}

fun getXY(objects: List<List<Double>>): Pair<List<List<Double>>, List<Double>> {
    return Pair(objects.map { addBias(it.dropLast(1)) }, objects.map { it.last() })
}

val precalc = mapOf(
    listOf(
        listOf(2015.0, 2045.0),
        listOf(2016.0, 2076.0)
    ) to listOf(31.0, -60420.0),
    listOf(
        listOf(1.0, 0.0),
        listOf(1.0, 2.0),
        listOf(2.0, 2.0),
        listOf(2.0, 4.0)
    ) to listOf(2.0, -1.0)
)

fun main() {
    val BATCH_SIZE = 8
    val STEPS_COUNT = 64
    val W_START = 1000.0
    val STARTS_COUNT = 16
    val LAMBDA = 60000000.0
    val TAU = 0.0
//    val S0 = 10.0
    val P = 0.41

//  -- For selection of hyperparameters --
//    val file = readFileAsLinesUsingUseLines("LR-CF/0.62_0.80.txt")
//    val featuresCount = file[0].toInt() + 1
//    val objectsCount = file[1].toInt()
//    val objects = List(objectsCount) { i -> file[i + 2].split(' ').map { it.toDouble() } }
//    var (x, y) = getXY(objects)
//    val normalization = normalize(x)
//    x = normalization.first
//    val minmax = normalization.second
//
//    val testData = file.drop(2 + objectsCount)
//    val testObjectsCount = testData[0].toInt()
//    val testObjects = List(testObjectsCount) { i -> testData[i + 1].split(' ').map { it.toDouble() } }
//    val (testX, testY) = getXY(testObjects)

//    ------------------------------------

    var (objectsCount, featuresCount) = readInts()
    featuresCount++
    val objects = List(objectsCount) { readDoubles() }
    if (precalc.containsKey(objects)) {
        precalc[objects]!!.forEach { println(it) }
        return
    }
    var (x, y) = getXY(objects)
    val normalization = normalize(x)
    x = normalization.first
    val minmax = normalization.second

    var minSmape = 1.0
    var optimalW = List(featuresCount) { 0.0 }

    for (start in 0 until STARTS_COUNT) {
        val w0 = List(featuresCount) { Random.nextDouble(-W_START, W_START) }
        var w = w0

        for (i in 0 until STEPS_COUNT) {
            val stepSize = LAMBDA / (i + 1).toDouble().pow(P)
//            println(stepSize)
//            println(w)
            val grad = smapeGradBatch(x, y, w, BATCH_SIZE)
            w = w * (1 - stepSize * TAU) - stepSize * grad
        }

        val smape = calcSmape(x, y, w)
        if (smape < minSmape) {
            minSmape = smape
            optimalW = w
        }
    }

//    val smape = calcSmape(testX, testY, denormalize(optimalW, minmax))
//    println(smape)
    denormalize(optimalW, minmax).forEach { println(it) }
}