#!/usr/bin/env python3
"""CLI entry point for Bank Parser Agent."""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import time

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import BankParserAgent


def setup_logging(verbose: bool = False):
    """Configure logging."""
    logger.remove()
    
    level = "DEBUG" if verbose else "INFO"
    
    # Console logging
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "agent.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )


def validate_inputs(pdf_path: Path, csv_path: Path, console: Console) -> bool:
    """Validate input files exist and are readable."""
    if not pdf_path.exists():
        console.print(f"[red]Error:[/red] PDF file not found at {pdf_path}")
        console.print(f"[yellow]Please create:[/yellow] {pdf_path}")
        return False
    
    if not csv_path.exists():
        console.print(f"[red]Error:[/red] CSV file not found at {csv_path}")
        console.print(f"[yellow]Please create:[/yellow] {csv_path}")
        return False
    
    # Check file sizes
    pdf_size = pdf_path.stat().st_size
    csv_size = csv_path.stat().st_size
    
    if pdf_size == 0:
        console.print(f"[red]Error:[/red] PDF file is empty")
        return False
    
    if csv_size == 0:
        console.print(f"[red]Error:[/red] CSV file is empty")
        return False
    
    return True


def display_banner(console: Console):
    """Display application banner."""
    banner_text = Text()
    banner_text.append("🤖 AI Bank Parser Agent\n", style="bold cyan")
    banner_text.append("Autonomous AI-powered bank statement parser generator", style="italic")
    
    banner = Panel(
        banner_text,
        border_style="cyan",
        padding=(1, 2)
    )
    
    console.print(banner)
    console.print()


def display_config(args, console: Console):
    """Display current configuration."""
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Target Bank", args.target.upper())
    table.add_row("Max Iterations", str(args.max_iter))
    table.add_row("Verbose Logging", "Yes" if args.verbose else "No")
    table.add_row("Model", args.model)
    table.add_row("Temperature", str(args.temperature))
    
    console.print(table)
    console.print()


def run_agent_with_progress(agent: BankParserAgent, bank_name: str, pdf_path: str, 
                           csv_path: str, output_path: str, console: Console) -> bool:
    """Run agent with rich progress display."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        # Create main task
        main_task = progress.add_task("[cyan]Generating parser...", total=100)
        
        # Start agent in a separate thread-like approach
        start_time = time.time()
        
        try:
            # Update progress based on agent status
            def update_progress():
                status = agent.get_status()
                progress.update(main_task, completed=status['progress'])
                
                # Update description based on status
                status_messages = {
                    'idle': '[cyan]Ready to start...',
                    'starting': '[yellow]Initializing agent...',
                    'analyzing': '[blue]Analyzing PDF and CSV...',
                    'generating': '[magenta]Generating parser code...',
                    'testing': '[green]Testing parser...',
                    'correcting': '[orange]Self-correcting code...',
                    'optimizing': '[purple]Optimizing performance...',
                    'completed': '[green]✅ Parser generated successfully!',
                    'failed': '[red]❌ Generation failed'
                }
                
                desc = status_messages.get(status['status'], status['status'])
                if status['iteration'] > 0:
                    desc += f" (Iteration {status['iteration']}/{status['max_iterations']})"
                
                progress.update(main_task, description=desc)
            
            # Run the agent
            success, error = agent.run(bank_name, pdf_path, csv_path, output_path)
            
            # Final update
            update_progress()
            
            execution_time = time.time() - start_time
            
            if success:
                progress.update(main_task, completed=100, 
                              description="[green]✅ Parser generated successfully!")
                console.print(f"\n[green]Success![/green] Parser generated in {execution_time:.1f}s")
                return True
            else:
                progress.update(main_task, description="[red]❌ Generation failed")
                console.print(f"\n[red]Failed![/red] {error}")
                return False
                
        except KeyboardInterrupt:
            progress.update(main_task, description="[red]❌ Cancelled by user")
            console.print(f"\n[yellow]Cancelled by user[/yellow]")
            return False
        except Exception as e:
            progress.update(main_task, description="[red]❌ Unexpected error")
            console.print(f"\n[red]Error:[/red] {str(e)}")
            return False


def display_results(success: bool, output_path: Path, console: Console):
    """Display final results."""
    if success:
        # Success panel
        success_text = Text()
        success_text.append("✅ Parser Generation Complete!\n\n", style="bold green")
        success_text.append(f"📄 Parser saved to: {output_path}\n", style="cyan")
        success_text.append("🧪 Run tests with: ", style="white")
        success_text.append("pytest tests/\n", style="yellow")
        success_text.append("🚀 Start web interface: ", style="white")
        success_text.append("python -m uvicorn api.main:app --reload", style="yellow")
        
        panel = Panel(
            success_text,
            title="🎉 Success",
            border_style="green",
            padding=(1, 2)
        )
        console.print(panel)
        
    else:
        # Failure panel
        failure_text = Text()
        failure_text.append("❌ Parser Generation Failed\n\n", style="bold red")
        failure_text.append("📋 Check logs at: ", style="white")
        failure_text.append("logs/agent.log\n", style="yellow")
        failure_text.append("🔧 Try adjusting parameters or check setup", style="white")
        
        panel = Panel(
            failure_text,
            title="💥 Failed",
            border_style="red",
            padding=(1, 2)
        )
        console.print(panel)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="AI Agent for generating bank statement parsers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py --target icici
  python agent.py --target sbi --max-iter 5 --verbose
  python agent.py --target hdfc --model llama-3.1-8b-instant
  python agent.py --target axis --temperature 0.2
        """
    )
    
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target bank name (e.g., icici, sbi, hdfc, axis, kotak)"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="Maximum self-correction iterations (default: 3)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama-3.3-70b-versatile",
        help="LLM model to use (default: llama-3.3-70b-versatile)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate setup without running agent"
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after successful generation"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    console = Console()
    
    # Display banner
    display_banner(console)
    
    # Paths
    bank_name = args.target.lower()
    data_dir = Path("data") / bank_name
    pdf_path = data_dir / f"{bank_name}_sample.pdf"
    csv_path = data_dir / f"{bank_name}_expected.csv"
    output_path = Path("custom_parsers") / f"{bank_name}_parser.py"
    
    # Display configuration
    display_config(args, console)
    
    # Validate input files
    if not validate_inputs(pdf_path, csv_path, console):
        sys.exit(1)
    
    # Initialize agent
    try:
        console.print("[cyan]Initializing AI agent...[/cyan]")
        agent = BankParserAgent(
            max_iterations=args.max_iter,
            model_name=args.model,
            temperature=args.temperature
        )
        console.print("[green]✓[/green] Agent initialized successfully")
        
    except Exception as e:
        console.print(f"[red]Failed to initialize agent:[/red] {e}")
        sys.exit(1)
    
    # Validate setup if requested
    if args.validate_only:
        console.print("\n[cyan]Validating setup...[/cyan]")
        issues = agent.validate_setup()
        
        if not issues:
            console.print("[green]✓ Setup validation passed![/green]")
            sys.exit(0)
        else:
            console.print("[red]Setup validation failed:[/red]")
            for issue in issues:
                console.print(f"  • {issue}")
            sys.exit(1)
    
    # Quick validation
    issues = agent.validate_setup()
    if issues:
        console.print("[yellow]⚠️  Setup issues detected:[/yellow]")
        for issue in issues:
            console.print(f"  • {issue}")
        console.print()
    
    # Run agent
    console.print(f"[bold]Starting parser generation for {bank_name.upper()}[/bold]\n")
    
    success = run_agent_with_progress(
        agent, bank_name, str(pdf_path), str(csv_path), str(output_path), console
    )
    
    # Display results
    console.print()
    display_results(success, output_path, console)
    
    # Run benchmark if requested and successful
    if success and args.benchmark:
        console.print("\n[cyan]Running benchmark...[/cyan]")
        try:
            from src.code_executor import CodeExecutor
            executor = CodeExecutor()
            
            with console.status("[cyan]Benchmarking parser performance..."):
                results = executor.benchmark_parser(str(output_path), str(pdf_path), 3)
            
            if "error" not in results:
                # Display benchmark results
                bench_table = Table(title="Benchmark Results", show_header=True)
                bench_table.add_column("Metric", style="cyan")
                bench_table.add_column("Value", style="green")
                
                bench_table.add_row("Average Time", f"{results['avg_time']:.2f}s")
                bench_table.add_row("Best Time", f"{results['min_time']:.2f}s")
                bench_table.add_row("Worst Time", f"{results['max_time']:.2f}s")
                bench_table.add_row("Success Rate", f"{results['success_rate']:.1f}%")
                bench_table.add_row("Iterations", str(results['iterations']))
                
                console.print(bench_table)
            else:
                console.print(f"[red]Benchmark failed:[/red] {results['error']}")
                
        except Exception as e:
            console.print(f"[red]Benchmark error:[/red] {e}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
