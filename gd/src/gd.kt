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

fun normalize(dataset: List<List<Double>>): List<List<Double>> {
    val featuresCount = dataset[0].size
    val minmax = Array(featuresCount) { Pair(0.0, 0.0) }
    for (feature in 0 until featuresCount) {
        val min = dataset.minBy { it[feature] }
        val max = dataset.maxBy { it[feature] }
        if (min == null || max == null) {
            return emptyList()
        }
        minmax[feature] = Pair(min[feature], max[feature])
    }

    val normalizedValues = List(dataset.size) { row ->
        List(featuresCount) { col ->
            val featureMinmax = minmax[col]
            val section = featureMinmax.second - featureMinmax.first
            if (col == featuresCount - 1) {
                dataset[row][col]
            } else if (section != 0.0) {
                (dataset[row][col] - featureMinmax.first) / section
            } else {
                1.0
            }
        }
    }

    return normalizedValues
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

fun smapeGradBatch(objs: List<List<Double>>, w: List<Double>, batchSize: Int): List<Double> {
    val batch: MutableSet<Int> = mutableSetOf()
    while (batch.size < batchSize) {
        batch.add((objs.indices).random())
    }

    var res = List(w.size) { 0.0 }

    for (i in batch) {
        val x = objs[i].dropLast(1)
        val a = w.dot(x)
        val y = objs[i].last()
        res = res + smapeGrad(x, w, y)
    }
    return res.map { it / batchSize }
}

fun calcSmape(objects: List<List<Double>>, w: List<Double>): Double {
    var sum = 0.0
    for (obj in objects) {
        sum += smape(obj.last(), w.dot(obj.dropLast(1)))
    }
    return sum / objects.size
}

fun addBias(values: List<Double>): List<Double> {
    val values = values.toMutableList()
    values.add(0, 1.0)
    return values
}

fun main() {
    val BATCH_SIZE = 200
    val STEPS_COUNT = 100
    val W_START = 1000000.0
    val STARTS_COUNT = 3
    val LAMBDA = 10000000000000.0
    val TAU = 1.0
    val S0 = 1.0
    val P = 0.35

//  -- For selection of hyperparameters --
    val file = readFileAsLinesUsingUseLines("LR-CF/0.40_0.65.txt")
    val featuresCount = file[0].toInt() + 1
    val objectsCount = file[1].toInt()
    var objects = List(objectsCount) { i -> addBias(file[i + 2].split(' ').map { it.toDouble() }) }
    objects = normalize(objects)

    val vData = file.drop(2 + objectsCount)
    val vObjectsCount = vData[0].toInt()
    var vObjects = List(vObjectsCount) { i -> addBias(vData[i + 1].split(' ').map { it.toDouble() }) }
    vObjects = normalize(vObjects)

//    ------------------------------------

//    var (objectsCount, featuresCount) = readInts()
//    featuresCount++
//    val objects = List(objectsCount) { addBias(readDoubles()) }


    var minSmape = 1.0
    var optimalW = List(featuresCount) { 0.0 }

    for (start in 0 until STARTS_COUNT) {
        val w0 = List(featuresCount) { Random.nextDouble(-W_START, W_START) }
        var w = w0

        for (i in 0 until STEPS_COUNT) {
            val stepSize =  LAMBDA * (S0 / (S0 + i)).pow(P)
            val grad = smapeGradBatch(objects, w, BATCH_SIZE)
//            println("${stepSize * grad}")
            w = w * (1 - stepSize * TAU) - stepSize * grad
        }

        val smape = calcSmape(objects, w)
        if (smape < minSmape) {
            minSmape = smape
            optimalW = w
        }
    }
    val smape = calcSmape(vObjects, optimalW)
    println(smape)
}