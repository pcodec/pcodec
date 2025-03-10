package io.github.pcodec;

/**
 * Configures how Pco should compress data.
 */
public class ChunkConfig {
    private int compressionLevel;

    /**
     * Returns ChunkConfig with the default compression level, ModeSpec, etc.
     */
    public ChunkConfig() {
        this.compressionLevel = 8;
    }

    public ChunkConfig withCompressionLevel(int compressionLevel) {
        this.compressionLevel = compressionLevel;
        return this;
    }
}
