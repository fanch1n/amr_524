import nox


@nox.session
def tests(session: nox.Session) -> None:
    session.install(".")
    session.install(".[test]")
    session.run("pytest", *session.posargs)
