#### Git Large File Storage (LFS)

LFS handles large files (.nc files) by storing references to the file in the repository, but not the actual file itself. To work around Git's architecture, Git LFS creates a pointer file which acts as a reference to the actual file (which is stored somewhere else). GitHub manages this pointer file in your repository. When you clone the repository, GitHub uses the pointer file as a map to go and find the large file for you.

1. Download and install the Git LFS:

[Mac - Intel Silicon](https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-darwin-amd64-v3.3.0.zip)

[Mac - Apple Silicon](https://github.com/git-lfs/git-lfs/releases/download/v3.3.0/git-lfs-darwin-arm64-v3.3.0.zip)

```bash0
sudo ./install.sh
```

Homebrew: ```bash brew install git-lfs```

MacPorts: ```bash port install git-lfs```

Once downloaded and installed, set up Git LFS for your user account by running:

```bash
git lfs install
```
You only need to run this once per user account.

2. In each Git repository where you want to use Git LFS, select the file types you'd like Git LFS to manage (or directly edit your .gitattributes). You can configure additional file extensions at anytime.

```bash
git lfs track "*.nc"
```

Now make sure .gitattributes is tracked:

```bash
git add .gitattributes
```

3. Commit and push to GitHub as you normally would; for instance, if your current branch is named main:

```bash
git add file.nc
git commit -m "Add nc file"
git push origin main
```

4. Git LFS can be disabled by running
```bash
git lfs uninstall
```
