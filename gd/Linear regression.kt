import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random
import smile.math.matrix.JMatrix

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

fun normalize(dataset: List<List<Double>>): List<List<Double>> {
    val featuresCount = dataset[0].size
    val minmax = MutableList(featuresCount) { Pair(0.0, 0.0) }
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

fun addBias(values: List<Double>): List<Double> {
    val values = values.toMutableList()
    values.add(0, 1.0)
    return values
}

fun getXY(obj: List<Double>): Pair<List<Double>, Double> {
    return Pair(obj.dropLast(1), obj.last())
}

fun nrmse(objects: List<List<Double>>, w: List<Double>): Double {
    var sum = 0.0
    var maxY = Double.MIN_VALUE
    var minY = Double.MAX_VALUE
    
    for (obj in objects) {
        val (x, y) = getXY(obj)
        maxY = max(maxY, y)
        minY = min(minY, y)
        val a = w.dot(x)
        sum += (a - y).pow(2.0)
    }
    var normalizer = maxY - minY
    if (normalizer == 0.0) {
        normalizer = if (maxY == 0.0) 1.0 else maxY
    }
    return sqrt(sum) / normalizer
}

fun sgd(train: List<List<Double>>, stepsCount: Int, step: Double, tau: Double): List<Double> {
    val featuresCount = train[0].size
    
    var w = List(featuresCount) { 0.0 }

    for (i in 0 until stepsCount) {
        val objInd = Random.nextInt(train.size)
        val (x, y) = getXY(train[objInd])
        val a = w.dot(x)
        val grad = (a - y) * x
        println("$a $y")
        w = w * (1 - step * tau) - step * grad
    }
    
    return w
}

fun readFileAsLinesUsingUseLines(fileName: String): List<String> = File(fileName).useLines { it.toList() }

fun main() {
    val inputFile = readFileAsLinesUsingUseLines("LR/2.txt")
    val featuresCount = inputFile[0].toInt() + 1
    val objectsCount = inputFile[1].toInt()
    var objects = List(objectsCount) { i -> addBias(inputFile[i + 2].split(' ').map { it.toDouble() }) }
    objects = normalize(objects)

    val testData = inputFile.drop(2 + objectsCount)
    val testObjectsCount = testData[0].toInt()
    var testObjects = List(testObjectsCount) { i -> addBias(testData[i + 1].split(' ').map { it.toDouble() }) }
    testObjects = normalize(testObjects)

    val STEPS_COUNT = 100
    val STEP_SIZE = 0.01
    val w = sgd(objects, STEPS_COUNT, STEP_SIZE, 0.75)
    val nrmse = nrmse(testObjects, w)
    println(nrmse)

    val F = objects.map { it.dropLast(1).toDoubleArray() }
    val FMatrix = JMatrix(F.toTypedArray())
    val svd = FMatrix.svd()
    val b = DoubleArray(objectsCount) { i -> objects[i].last() }
    var solution = DoubleArray(featuresCount)
    svd.solve(b, solution)
    println(solution.toList())

    println(nrmse(testObjects, solution.toList()))
}