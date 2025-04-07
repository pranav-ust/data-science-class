# GitHub Practice Repository

Welcome to the GitHub Practice repository! This repository is designed to help you learn the basics of GitHub collaboration, including forking, editing files, committing changes, and creating pull requests.

## Exercise Instructions

In this exercise, you will:
1. Fork this repository
2. Add your name to the list of contributors in this README
3. Commit your changes
4. Create a pull request
5. Have your pull request approved and merged

### Prerequisites

- A GitHub account (create one at [github.com](https://github.com) if you don't have one)
- For command line users: Git installed on your computer ([installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git))

## Detailed Instructions

### Using the GitHub Web Interface

#### Step 1: Fork the Repository
1. Click the "Fork" button at the top right of this page
2. This creates a copy of the repository under your GitHub account

#### Step 2: Edit the README.md File
1. In your forked repository, click on the `README.md` file
2. Click the pencil icon (Edit this file) at the top right of the file content
3. Scroll down to the "Contributors" section
4. Add your name to the list following this format: `- [Your Name]`
5. Scroll to the bottom of the page

#### Step 3: Commit Your Changes
1. Under "Commit changes," enter a commit message like "Add [your name] to contributors list"
2. Select "Commit directly to the main branch"
3. Click "Commit changes"

#### Step 4: Create a Pull Request
1. After committing, you'll return to your repository
2. Click on "Pull requests" in the top navigation
3. Click the green "New pull request" button
4. Ensure the base repository is set to the original repository (instructor's) and the head repository is your fork
5. Click "Create pull request"
6. Enter a title for your pull request (e.g., "Add [your name] to contributors")
7. Optionally add a description
8. Click "Create pull request"

### Using the Command Line

#### Step 1: Fork the Repository
1. Click the "Fork" button at the top right of this page in your browser
2. This creates a copy of the repository under your GitHub account

#### Step 2: Clone Your Fork Locally
```bash
# Replace YOUR-USERNAME with your GitHub username
git clone https://github.com/YOUR-USERNAME/github-practice.git
cd github-practice
```

#### Step 3: Edit the README.md File
Open the README.md file in your preferred text editor and add your name to the Contributors section.

```bash
# Example using nano (you can use any text editor)
nano README.md
```

Add your name in the format: `- [Your Name]` to the Contributors section.

#### Step 4: Commit and Push Your Changes
```bash
# Stage your changes
git add README.md

# Commit with a descriptive message
git commit -m "Add [your name] to contributors list"

# Push to your fork
git push origin main
```

#### Step 5: Create a Pull Request
1. Go to your fork on GitHub in your web browser
2. You should see a notification about your recent push, with an option to create a pull request
3. Click on "Compare & pull request"
4. Ensure the base repository is the original (instructor's) and the head repository is your fork
5. Add a title like "Add [your name] to contributors"
6. Click "Create pull request"

## What Happens Next?

After you create a pull request:
1. The instructor will be notified
2. They may review your changes and leave comments
3. If everything looks good, they'll approve and merge your pull request
4. Your name will appear in the main repository's README!

## Troubleshooting

- Make sure you're editing your fork, not the original repository
- If you don't see your changes, refresh the page
- If you have issues with the command line, try the web interface instead
- If your pull request shows conflicts, you may need to sync your fork first

## Contributors

- Pranav A
<!-- Add your name below this line -->
- Aaron

