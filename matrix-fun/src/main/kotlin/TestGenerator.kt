import kotlin.random.Random

fun main() {
    val n = Random.nextInt(1, 50)
    val m = Random.nextInt(1, n)
    val k = Random.nextInt(1, n)
    println("$n $m $k")
    for (i in 0 until n) {
        val type = Random.nextInt(1, 6)
        when (type) {
            1 -> {
                println("var")
            }
        }
    }
}