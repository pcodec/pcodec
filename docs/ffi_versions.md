Pco versions for each language can change independently; they are governed only
by API compatibility in the language of the package.

Rust is the implementation that other languages bind to. If another language
uses Rust version `a.b.c` in one of these tables, then it is really using code
from somewhere in between `a.b.c` whatever Rust version follows that.

# Python versions

| Python `pcodec` version | Rust `pco` version |
|-------------------------|--------------------|
| 0.0.0                   | 0.1.3              |
| 0.0.1                   | 0.1.3              |
| 0.1.0                   | 0.1.3              |
| 0.1.1                   | 0.2.5              |
| 0.2.0                   | 0.3.0              |
| 0.3.0                   | 0.4.0-alpha        |
| 0.3.1                   | 0.4.0              |
| 0.3.2                   | 0.4.1              |
| 0.3.3                   | 0.4.2              |
| 0.3.4                   | 0.4.4              |

# Java / JVM versions

| Java `io.github.pcodec.pco-jni` version | Rust `pco` version |
|-----------------------------------------|--------------------|
| 0.1.0                                   | 0.4.2              |
| 0.1.1                                   | 0.4.4              |

