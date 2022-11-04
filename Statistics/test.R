# Plot a normal distribution:
x <- seq(-4, 4, length.out = 100)
y <- dnorm(x)
plot(x, y, type = "l", lwd = 2, xlab = "x", ylab = "density")
