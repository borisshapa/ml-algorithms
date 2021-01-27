import java.io.File
import java.lang.StringBuilder
import kotlin.random.Random

fun main() {
    val writer = File("output.txt").bufferedWriter()
    val m = 100
    val k = 20
    val h = 10
    writer.write("$m $k $h\n")

    val n = 2000
    writer.write("$n\n")

    for (i in 0 until n) {
        val str = StringBuilder()
        for (j in 0 until m) {
            val feature = Random.nextInt(-1000_000_000, 1000_000_000)
            str.append("$feature ")
        }
        val cl = Random.nextInt(1, k)
        str.append("$cl\n")
        writer.write(str.toString())
    }
}