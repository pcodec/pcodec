# The Go package links the Rust cpcodec static library; build it first.

.PHONY: rust test bench vet

rust:
	cargo build --release -p cpcodec

test: rust
	go test ./...

bench: rust
	go test -run=^$$ -bench=. -benchmem ./...

vet:
	go vet ./...
