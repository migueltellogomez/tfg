from gtpinterface import gtpinterface
from imitationagent import imitationagent
def main():
	"""
	Main function, simply sends user input on to the gtp interface and prints
	responses.
	"""
	agent = imitationagent()
	interface = gtpinterface(agent)
	while True:
		command = input()
		success, response = interface.send_command(command)
		print(("= " if success else "? ")+response+'\n')
		success, response = interface.send_command("showboard")
		print(response)
if __name__ == "__main__":
	main()
