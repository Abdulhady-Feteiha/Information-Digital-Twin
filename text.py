
def encode(taxi_row, taxi_col, pass_loc, dest_idx):
        # (15) 15, 9, 8
        i = taxi_row
        i *= 15
        i += taxi_col
        i *= 9
        i += pass_loc
        i *= 8
        i += dest_idx
        return i

def decode(i):
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 10)
        i = i // 10
        out.append(i)
        assert 0 <= i < 10
        return out

def encode(taxi_row, taxi_col, pass_loc, dest_idx):
        return ((taxi_row * 10 + taxi_col) * 5 + pass_loc) * 4 + dest_idx

print(encode(9,9,4,3))
print(decode(1047))