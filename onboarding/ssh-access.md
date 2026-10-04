# SSH into the robot

1. Connect your laptop to the same network as the robot.
2. Copy the current robot IP from [this gist](https://gist.github.com/naowalrahman/f9a3c308463b22afbf643dae6bef981a).
3. Open a terminal on your laptop and replace `ROBOT_IP` below:

   ```bash
   ssh curc@ROBOT_IP
   ```

4. On your first connection, verify the host fingerprint with the team and type `yes` if it matches.
5. Enter password `curc` when prompted (typing will not appear).

You are now in a terminal on the robot.
Run `exit` to disconnect.
