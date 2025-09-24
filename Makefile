.PHONY: backtest trade lint

backtest:
	python btc_bot.py backtest

trade:
	python btc_bot.py trade

lint:
	ruff .
