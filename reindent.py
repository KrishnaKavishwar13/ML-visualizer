with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out = []
in_if = False
for line in lines:
    if line.startswith('if uploaded_file is not None:'):
        in_if = True
        out.append(line)
        out.append('    data_pane = st.container()\n')
        out.append('    with data_pane:\n')
        out.append('        st.markdown("""<style>div[data-testid="stVerticalBlock"]:has(> div > div > #csv-marker) { background: rgba(15, 23, 42, 0.8) !important; padding: 40px !important; border-radius: 20px !important; box-shadow: 0 8px 25px rgba(0,0,0,0.5); z-index: 5; position: relative; }</style><div id="csv-marker"></div>""", unsafe_allow_html=True)\n')
        continue
    
    if in_if:
        if line.startswith('else:'):
            in_if = False
            out.append(line)
        elif line.strip() == '':
            out.append(line)
        elif line.startswith('    ') and not line.startswith('        ') and not line.startswith('    # CSS pane applied via container'):
            # It's an indented line under 'if', so add 4 spaces
            out.append('    ' + line)
        elif line.startswith('        '):
             out.append('    ' + line)
        elif line.startswith('    # CSS pane'):
             pass # Drop it
        else:
             out.append(line)
    else:
        out.append(line)

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(out)
