import sys

def unindent_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    out = []
    in_if = False
    skip_next = False
    for i, line in enumerate(lines):
        if 'color: white' in line or 'color: #f5f5f5' in line:
            if 'stButton' not in line and 'Sidebar' not in line and 'grid-effect' not in line:
                line = line.replace('color: white', 'color: black').replace('color: #f5f5f5', 'color: black')
        
        if line.startswith('if uploaded_file is not None:'):
            in_if = True
            out.append(line)
            continue
        
        if in_if:
            if 'data_pane = st.container()' in line:
                continue
            if 'with data_pane:' in line:
                continue
            if '<style>div[data-testid="stVerticalBlock"]:has(' in line:
                continue
            if line.startswith('else:'):
                in_if = False
                out.append(line)
                continue
                
            if line.startswith('    '):
                # remove exactly one level of indentation (4 spaces)
                out.append(line[4:])
            else:
                out.append(line)
        else:
            out.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(out)

unindent_file('app.py')
unindent_file('app3.py')
