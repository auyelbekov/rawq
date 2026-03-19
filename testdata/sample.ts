interface Logger {
    log(message: string): void;
    error(message: string): void;
}

class ConsoleLogger implements Logger {
    private prefix: string;

    constructor(prefix: string) {
        this.prefix = prefix;
    }

    log(message: string): void {
        console.log(`${this.prefix}: ${message}`);
    }

    error(message: string): void {
        console.error(`${this.prefix} ERROR: ${message}`);
    }
}

function createLogger(prefix: string): Logger {
    return new ConsoleLogger(prefix);
}
