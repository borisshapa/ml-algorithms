import kotlin.collections.ArrayList
import kotlin.math.log2

private fun readLn() = readLine()!!
private fun readInt() = readLn().toInt()
private fun readLong() = readLn().toLong()
private fun readDouble() = readLn().toDouble()
private fun readStrings() = readLn().split(" ")
private fun readInts() = readStrings().map { it.toInt() }
private fun readLongs() = readStrings().map { it.toLong() }
private fun readDoubles() = readStrings().map { it.toDouble() }

class Object(val x: DoubleArray, val y: Int)

class DecisionTree(private val maxDepth: Int = 5, private val classesCount: Int) {
    lateinit var root: Node
    lateinit var criterion: Criterion

    fun fit(x: List<DoubleArray>, y: List<Int>) {
        criterion = if (x.size < 200) Criterion.ENTROPY else Criterion.GINI
        val data = x.zip(y).map { Object(it.first, it.second) }
        root = trainNode(data, 0)
    }

    private fun impurity(classesStat: Array<Int>, size: Int): Double {
        return classesStat.sumByDouble {
            if (it == 0) {
                0.0
            } else {
                val p = it.toDouble() / size
                if (criterion == Criterion.GINI) {
                    p * (1.0 - p)
                } else {
                    p * log2(1.0 / p)
                }
            }
        }
    }

    private fun trainNode(data: List<Object>, depth: Int): Node {
        val classesStat = data.groupingBy { it.y }.eachCount()
        if (classesStat.size <= 1 || depth >= maxDepth) {
            return LeafNode(classesStat.maxByOrNull { it.value }!!.key)
        }
        val featureCount = data.first().x.size

        var maxQuality = -Double.MAX_VALUE
        var splitThreshold = 0.0
        var splitFeature = -1

        for (feature in 0 until featureCount) {
            val sortedObjs = data.sortedBy { it.x[feature] }
            val sampleSize = sortedObjs.size

            val leftClasses = Array(classesCount + 1) { 0 }
            val rightClasses = Array(classesCount + 1) { 0 }
            sortedObjs.forEach { rightClasses[it.y]++ }

            val fullImpurity = impurity(rightClasses, sampleSize)

            var prevThreshold = sortedObjs.first().x[feature]
            for ((ind, obj) in sortedObjs.withIndex()) {
                val threshold = obj.x[feature]

                if (threshold != prevThreshold) {
                    val p = ind.toDouble() / sampleSize
                    val quality = fullImpurity -
                            p * impurity(leftClasses, ind) -
                            (1.0 - p) * impurity(rightClasses, sampleSize - ind)

                    if (quality > maxQuality) {
                        maxQuality = quality
                        splitFeature = feature
                        splitThreshold = (threshold + prevThreshold) / 2
                    }
                    prevThreshold = threshold
                }
                rightClasses[obj.y]--
                leftClasses[obj.y]++
            }
        }

//        println("$maxQuality $splitFeature $splitThreshold")

        val (left, right) = data.partition { it.x[splitFeature] < splitThreshold }
        val leftNode = trainNode(left, depth + 1)
        val rightNode = trainNode(right, depth + 1)
        return SplitNode(splitFeature, splitThreshold, leftNode, rightNode)
    }

    open class Node {
        var id: Int = -1
    }

    class LeafNode(val prediction: Int) : Node()

    class SplitNode(val splitFeature: Int, val splitThreshold: Double, val left: Node, val right: Node) : Node()

    enum class Criterion {
        GINI,
        ENTROPY
    }
}

var freeId = 1

fun dfsInit(node: DecisionTree.Node) {
    node.id = freeId++
    if (node is DecisionTree.SplitNode) {
        dfsInit(node.left)
        dfsInit(node.right)
    }
}

fun dfsPrint(node: DecisionTree.Node) {
    if (node is DecisionTree.LeafNode) {
        println("C ${node.prediction}")
    }
    if (node is DecisionTree.SplitNode) {
        println("Q ${node.splitFeature + 1} ${node.splitThreshold} ${node.left.id} ${node.right.id}")
        dfsPrint(node.left)
        dfsPrint(node.right)
    }
}

fun main() {
    val (_, k, maxDepth) = readInts()
    val n = readInt()

    val x = ArrayList<DoubleArray>()
    val y = ArrayList<Int>()

    for (i in 0 until n) {
        val input = readDoubles()
        y.add(input.last().toInt())
        x.add(input.dropLast(1).toDoubleArray())
    }

    val dt = DecisionTree(maxDepth, k)
    dt.fit(x, y)
    dfsInit(dt.root)
    println(freeId - 1)
    dfsPrint(dt.root)
}