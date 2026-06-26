import re
import os

emojis_to_remove = ['🧠', '📂', '📄', '📊', '🧹', '⚙️', '📈', '✅', '❌', '⚠️', '🎯', '✨']

def remove_emojis_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        
    for emoji in emojis_to_remove:
        content = content.replace(emoji + ' ', '')
        content = content.replace(emoji, '')
        
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

remove_emojis_from_file('app.py')
remove_emojis_from_file('app3.py')
