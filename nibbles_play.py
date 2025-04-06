import sys
import time
import pygame
import torch
from rich.console import Console
from nibbles_train import DQNAgent
from nibbles_env import NibblesEnv

# ───────────────────────────────────────── CONSTANTS ───────────────────────────────────────── #
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
FONT_NAME = "Arial"
FONT_SIZE = 24
MAX_LEVEL = 10
SCORE_THRESHOLD = 3
MODEL_PATH = "trained_model/dqn_snake_final_20250404-230508.pth"
# ───────────────────────────────────────────────────────────────────────────────────────────── #
console = Console()

def play_trained_model(model_path: str, render_mode: bool = True) -> None:
    """
    Plays the Nibbles game using a pre-trained DQN model.

    Args:
        model_path (str): Path to the trained model file.
        render_mode (bool): Whether to render the game visually or not.

    Returns:
        None
    """
    try:
        pygame.init()
        font = pygame.font.SysFont(FONT_NAME, FONT_SIZE)
        level = 0
        agent = DQNAgent(
            state_dim=21,
            action_dim=4,
            lr=3e-4,
            gamma=0.99,
            epsilon=0.01,
            epsilon_min=0.01,
            epsilon_decay=1.0,
            batch_size=256,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            target_update_freq=1000,
            prioritized_replay=True,
            double_dqn=True
        )

        agent.q_net.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        agent.epsilon = agent.epsilon_min

        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Trained Nibbles Agent")

        def exit_note() -> bool:
            """Display a 'play again' message and handle user input."""
            screen.fill((0, 0, 0))
            text = font.render("Well Done! Wanna play again? [Y/N]", True, (255, 255, 255))
            screen.blit(text, (160, 220))
            pygame.display.flip()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_y:
                            return False
                        elif event.key == pygame.K_n:
                            pygame.quit()
                            sys.exit()

        def ask_exit_confirmation() -> None:
            """Prompt the user to confirm exit."""
            screen.fill((0, 0, 0))
            text1 = font.render("Do you want to exit? [Y/N]", True, (255, 255, 255))
            screen.blit(text1, (160, 220))
            pygame.display.flip()

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_y:
                            pygame.quit()
                            sys.exit()
                        elif event.key == pygame.K_n:
                            return

        while True:
            env = NibblesEnv(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, level=level, render_mode=render_mode)
            state = env.reset()
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            ask_exit_confirmation()
                        elif event.key == pygame.K_UP and level < MAX_LEVEL:
                            level += 1
                            console.print(f"[yellow]Manually increased level to {level}[/yellow]")
                            break
                        elif event.key == pygame.K_DOWN and level > 0:
                            level -= 1
                            console.print(f"[yellow]Manually decreased level to {level}[/yellow]")
                            break
                print(f'LEVEL: {level}')
                if level < MAX_LEVEL and env.score >= SCORE_THRESHOLD:
                    level += 1
                    env.score = 0
                    console.print(f"\n[cyan]--- Auto-increasing obstacle level to {level} ---[/cyan]")
                    break

                if level == MAX_LEVEL and env.score >= SCORE_THRESHOLD:
                    console.print("\n[bold green]--- Game finished: Level 10 completed! Well done! ---[/bold green]")
                    done = exit_note()
                    if not done:
                        level, env.score = 0, 0

                action = agent.select_action(state)
                state, reward, done, _ = env.step(action)

                if render_mode:
                    env.render([], [])

            time.sleep(1)

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    play_trained_model(MODEL_PATH, render_mode=True)
