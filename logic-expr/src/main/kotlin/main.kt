import kotlin.math.log2

private fun readLn() = readLine()!!
private fun readInt() = readLn().toInt()
private fun readLong() = readLn().toLong()
private fun readDouble() = readLn().toDouble()
private fun readStrings() = readLn().split(" ")
private fun readInts() = readStrings().map { it.toInt() }
private fun readLongs() = readStrings().map { it.toLong() }
private fun readDoubles() = readStrings().map { it.toDouble() }

fun cnf(a: Array<Int>, varCount: Int) {
    var zeroResCount = 0
    for ((ind, res) in a.withIndex()) {
        if (res == 0) {
            zeroResCount++
            val list = mutableListOf<Double>()
            var cntZero = 0
            for (i in varCount - 1 downTo 0) {
                if ((ind and (1 shl i)) == 0) {
                    list.add(-1.0)
                    cntZero++
                } else {
                    list.add(1.0)
                }
            }
            list.add(-varCount + cntZero + 0.1)
            list.forEach { print("$it ") }
            println()
        }
    }
    for (i in 0 until zeroResCount) {
        print("-1.0 ")
    }
    println("0.1")
}

fun dnf(a: Array<Int>, varCount: Int) {
    var oneResCount = 0
    for ((ind, res) in a.withIndex()) {
        if (res == 1) {
            oneResCount++
            val list = mutableListOf<Double>()
            var cntOne = 0
            for (i in varCount - 1 downTo 0) {
                if ((ind and (1 shl i)) == 0) {
                    list.add(-1.0)
                } else {
                    list.add(1.0)
                    cntOne++
                }
            }
            list.add(-cntOne + 0.1)
            list.forEach { print("$it ") }
            println()
        }
    }
    for (i in 0 until oneResCount) {
        print("1.0 ")
    }
    println("-0.1")
}

fun main() {
    val n = readInt()
    val arr = Array(1 shl n) { 0 }
    for (i in 0 until (1 shl n)) {
        arr[i] = readInt()
    }

    val zeroCount = arr.count { it == 0 }
    val oneCount = arr.size - zeroCount
    if (zeroCount == 0) {
        println("1\n1")
        for (i in 0 until n) {
            print("0.0 ")
        }
        println("0.1")
        return
    }

    if (oneCount == 0) {
        println("1\n1")
        for (i in 0 until n) {
            print("0.0 ")
        }
        println("-0.1")
        return
    }

    println(2)
    if (zeroCount < oneCount) {
        println("$zeroCount 1")
        cnf(arr, n)
    } else {
        println("$oneCount 1")
        dnf(arr, n)
    }
}