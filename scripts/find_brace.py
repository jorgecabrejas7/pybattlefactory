
def find_unbalanced_brace(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    balance = 0
    # Start checking from 953 (Switch start)
    # Actually, check from start of function (910)
    start_line = 910
    
    # We need to account for balance at start of function (0 inside function body)
    # Function start: { at 910.
    
    print(f"Checking {filename} starting from line {start_line}...")
    
    for i, line in enumerate(lines):
        lineno = i + 1
        if lineno < start_line:
            continue
            
        # Strip comments (simple)
        code = line.split('//')[0]
        
        for char in code:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1
                if balance < 0: # Assuming we started at function scope 0? No, usually function has {
                    # But finding absolute imbalance is tricky if we don't start from 0.
                    # Let's just track balance change from line 952.
                    pass

    # New approach: Track balance relative to line 953 (Switch start).
    # Switch start line 953 has `{`.
    # Prior to that, we are in function body (Balance 1).
    # At 953 end, Balance 2.
    
    # Let's run from 910.
    curr_balance = 0
    
    for i, line in enumerate(lines):
        lineno = i + 1
        if lineno < 910: continue
        
        code = line.split('//')[0]
        for char in code:
            if char == '{':
                curr_balance += 1
            elif char == '}':
                curr_balance -= 1
        
        if curr_balance == 1:
            print(f"Switch closed at line {lineno} via brace!")
            
        # We expect curr_balance to be >= 1 (Function body) until the end.
        if curr_balance == 0:
             print(f"Function closed at line {lineno} via brace!")
             # If this is before 1360, that's the problem.
             
        if curr_balance < 0:
             print(f"IMBALANCE: Negative balance at line {lineno}")
             return

find_unbalanced_brace('src/battle_engine.cpp')
