package sample

import "math"

// Fibonacci returns the nth Fibonacci number.
func Fibonacci(n int) int {
	if n <= 1 {
		return n
	}
	a, b := 0, 1
	for i := 2; i <= n; i++ {
		a, b = b, a+b
	}
	return b
}

// IsPrime checks if a number is prime.
func IsPrime(n int) bool {
	if n < 2 {
		return false
	}
	for i := 2; i <= int(math.Sqrt(float64(n))); i++ {
		if n%i == 0 {
			return false
		}
	}
	return true
}

// GCD computes greatest common divisor using Euclidean algorithm.
func GCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}
