# Step 1: Add upstream (first time only)
git remote add upstream https://github.com/leejet/stable-diffusion.cpp.git

# Step 2: Fetch all branches
git fetch upstream
git fetch origin

# Step 3: Checkout your main branch
git checkout master  # or main

# Step 4: Create backup of your work
git branch backup-my-changes

# Step 5: Rebase your commits on top of upstream
git rebase upstream/master

# Step 6: If conflicts occur, resolve them:
# - Edit conflicting files
# - Stage resolved files: git add <file>
# - Continue: git rebase --continue
# - Or abort: git rebase --abort

# Step 7: Push changes (force push needed after rebase)
git push origin master --force-with-lease
