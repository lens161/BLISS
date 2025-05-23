import os

def tail(filename, lines=20):
    with open(filename, 'rb') as f:
        f.seek(0, os.SEEK_END)
        end_pos = f.tell()
        buffer = bytearray()
        newline_count = 0

        # Start reading backwards
        for i in range(end_pos - 1, -1, -1):
            f.seek(i)
            byte = f.read(1)
            buffer.insert(0, byte[0])  # prepend byte to buffer
            if byte == b'\n':
                newline_count += 1
                if newline_count == lines + 1:  # +1 because the last line may not end with '\n'
                    break

        return buffer.decode(errors='replace').splitlines()[-lines:]

# Example usage
last_20_lines = tail('outputs/job.48066.out', 20)
for line in last_20_lines:
    print(line)