# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

         # User specific aliases and functions
alias _emacs="emacs -nw"
alias _chmod_X="chmod +x"
alias _clear_emacs="rm *~ ; rm \#*\#"
alias _pep8="pycodestyle-3"
alias _pwd="pwd |tr -d '\n'  | xclip -selection c"
#diplay file with according color type.
alias _cat='highlight'
#display json file clearly:
alias _prettyjson='python -m json.tool'

#Split Window to 4 and manage the access to each one
alias _SplitAndShit='bash ~/Programming/Holberton/Helpfull_Scripts/XMACRO/SplitAndShit.sh'
#Prepare necessary envirement here ..
alias _prepareEnvi='mkdir ToBeprint ; mkdir -p Testing/Screenshots/ ; cp ~/Programming/Holberton/Helpfull_Scripts/pushFile.sh . && cp -r  ~/Programming/Holberton/Helpfull_Scripts/Revise/ Testing && cp ~/Programming/Holberton/Helpfull_Scripts/Rename_By_regex.sh Testing/Screenshots'
         #customize your Prompt

   #colors Variable:
   RCol='\033[0m'
   Gre='\033[32m';
   Red='\033[31m';
   Blu='\033[34m';
   Yel='\033[33m';

   #Parent & Current Directory:
   P_C='$(basename $(dirname "$PWD"))/$(basename "$PWD")'

   #PS1 costumized
   PS1="${RCol}┌─[\`if [ \$? = 0 ]; then echo "${Gre}"; else echo "${Red}"; fi\`\t\[${Rcol}\] \[${Blu}\]\u\[${RCol}\] \[${Yel}\]\[${P_C}\]\[${RCol}\]]\n└─▶ "

   #pyenv see https://github.com/pyenv/pyenv#basic-github-checkout
   # 2.Configure your shell's environment for Pyenv
   eval "$(pyenv init -)"

   # Environment for Dart lib-> check the org file-> ~/Programming/Dart/ORG/Dart.org
   export PATH="$PATH:/home/cuore-pc/Programming/Dart/Archive/dart-sdk/bin"
