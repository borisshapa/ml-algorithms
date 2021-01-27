import java.util.*
import kotlin.math.exp
import kotlin.math.ln

private fun readLn() = readLine()!!
private fun readInt() = readLn().toInt()
private fun readLong() = readLn().toLong()
private fun readDouble() = readLn().toDouble()
private fun readStrings() = readLn().split(" ")
private fun readInts() = readStrings().map { it.toInt() }
private fun readLongs() = readStrings().map { it.toLong() }
private fun readDoubles() = readStrings().map { it.toDouble() }

class NaiveBayes(val alpha: Double, val lambdas: DoubleArray, val nClasses: Int) {
    var pWordY = listOf<MutableList<Double>>()
    var pY = mutableListOf<Double>()

    fun fit(x: List<IntArray>, y: List<Int>) {
        dictSize = freeId
        val classes = mutableMapOf<Int, MutableList<IntArray>>()
        for (cl in 0 until nClasses) {
            classes[cl] = mutableListOf()
        }
        y.forEachIndexed { ind, cl ->
            classes[cl - 1]!!.add(x[ind])
        }
//        classes.forEach {
//            println(it.key)
//            for (arr in it.value) {
//                arr.forEach { el -> print("$el ") }
//                println()
//            }
//            println()
//        }
        pWordY = List(nClasses) { MutableList(dictSize) { 0.0 } }
        pY = MutableList(nClasses) { 0.0 }

        for ((cl, sample) in classes) {
            val nSentencesWithWord = Array(dictSize) { 0 }
            for (sentence in sample) {
                for (word in 0 until dictSize) {
                    if (sentence[word] != 0) {
                        nSentencesWithWord[word]++
                    }
                }
            }

            for (word in 0 until dictSize) {
                pWordY[cl][word] = (nSentencesWithWord[word] + alpha) / (sample.size + alpha * 2)
            }
            pY[cl] = sample.size.toDouble() / y.size
        }
    }

    fun predict(x: IntArray): List<Double> {
        val risks = DoubleArray(nClasses) { 0.0 }
        for (cl in 0 until nClasses) {
            var risk = ln(lambdas[cl]) + ln(pY[cl])
            for (word in 0 until dictSize) {
                risk += ln(if (x[word] != 0) pWordY[cl][word] else 1 - pWordY[cl][word])
            }
            risks[cl] = risk
        }
        val max = risks.maxOrNull()!!
        val exps =  risks.map { exp(it - max) }
        val sum = exps.sum()
        return exps.map { it / sum }
    }
}

var freeId = 0
var dictSize = 0
val stringToId = mutableMapOf<String, Int>()

fun stringListToIdList(strs: List<String>): List<Int> {
    return List(strs.size) {
        val str = strs[it]
        if (stringToId.containsKey(str)) {
            stringToId[str]!!
        } else {
            stringToId[str] = freeId
            freeId++
        }
    }
}

fun wordStat(x: List<List<Int>>): List<IntArray> {
    return List(x.size) {
        val wordStat = IntArray(dictSize) { 0 }
        for (word in x[it]) {
            if (word < dictSize) {
                wordStat[word]++
            }
        }
        wordStat
    }
}

fun main() {
    val nClasses = readInt()
    val lambdas = readDoubles()
    val alpha = readDouble()
    val n = readInt()

    val y = mutableListOf<Int>()
    val x = mutableListOf<List<Int>>()
    for (i in 0 until n) {
        val input = readStrings()
        y.add(input.first().toInt())
        x.add(stringListToIdList(input.drop(2)))
    }
    dictSize = freeId

    val xWordStat = wordStat(x)

    val naiveBayes = NaiveBayes(alpha, lambdas.toDoubleArray(), nClasses)
    naiveBayes.fit(xWordStat, y)

    val m = readInt()
    val test = List(m) {
        val input = readStrings()
        stringListToIdList(input.drop(1))
    }

    val testWordStat = wordStat(test)
    for (sentence in testWordStat) {
        val pred = naiveBayes.predict(sentence)
        pred.forEach { print("$it ") }
        println()
    }
}