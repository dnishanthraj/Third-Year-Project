# Set variables
REMOTE=origin  # Adjust if using a different remote
BRANCH=$(git rev-parse --abbrev-ref HEAD)  # Get current branch
BATCH_SIZE=500  # Adjust batch size if necessary (500 recommended)

# Check if the branch exists on the remote
if git show-ref --quiet --verify refs/remotes/$REMOTE/$BRANCH; then
    # If the branch exists on the remote, only push missing commits
    range=$REMOTE/$BRANCH..HEAD
else
    # If the branch doesn't exist on the remote, push all commits
    range=HEAD
fi

# Count the number of commits to push
n=$(git log --first-parent --format=format:x $range | wc -l)

# Push commits in batches
for i in $(seq $n -$BATCH_SIZE 1); do
    # Find the hash of the next commit batch
    h=$(git log --first-parent --reverse --format=format:%H --skip $i -n1)
    echo "Pushing commit batch ending at $h..."
    # Push the commit batch
    git push $REMOTE ${h}:refs/heads/$BRANCH
done

# Push the final batch of commits (if any remain)
git push $REMOTE HEAD:refs/heads/$BRANCH
